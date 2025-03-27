import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import hydra
import omegaconf as OmegaConf
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from models.unet import UNet
from ame_furi.ddpm import DDPM, LinearNoiseScheduler

def setup_logging(cfg, rank):
    """Setup logging and tensorboard writer"""
    if rank != 0:
        return None
    
    # Create directories if they don't exist
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.sample_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path(cfg.logging.log_dir) / "training.log"),
            logging.StreamHandler()
        ]
    )
    writer = SummaryWriter(cfg.logging.tensorboard_dir)
    return writer

def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    logging.info(f"Loaded checkpoint from epoch {epoch} with loss {best_loss:.4f}")
    return epoch, best_loss

@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(cfg):
    # Initialize distributed training if needed
    rank = 0
    if cfg.distributed:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    
    # Setup logging and directories
    writer = setup_logging(cfg, rank)
    
    # Set random seeds for reproducibility
    torch.manual_seed(cfg.training.seed + rank)
    np.random.seed(cfg.training.seed + rank)
    
    # Initialize model
    model = UNet(
        n_channels=cfg.model.in_channels,
        n_classes=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        depth=cfg.model.depth,
        time_emb_dim=cfg.model.time_emb_dim
    ).to(rank)
    
    if cfg.distributed:
        model = DDP(model, device_ids=[rank])
    
    # Initialize DDPM components
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=cfg.diffusion.num_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end
    )
    
    ddpm = DDPM(model, noise_scheduler, device=f"cuda:{rank}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=cfg.optimizer.min_lr
    )
    
    # Data loaders
    train_loader, val_loader, train_sampler = get_mnist_loaders(
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        distributed=cfg.distributed
    )
    
    # Training state
    start_epoch = 0
    best_loss = float('inf')
    
    # Load checkpoint if exists
    checkpoint_path = Path(cfg.logging.checkpoint_dir) / "latest_checkpoint.pth"
    if cfg.training.resume and checkpoint_path.exists():
        start_epoch, best_loss = load_checkpoint(
            checkpoint_path,
            model.module if cfg.distributed else model,
            optimizer,
            scheduler,
            f"cuda:{rank}"
        )
        start_epoch += 1  # Start from next epoch
    
    # Training loop
    for epoch in range(start_epoch, cfg.training.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(rank)
            
            with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
                loss = ddpm.train_step(images, optimizer)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if cfg.training.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            if rank == 0 and batch_idx % cfg.logging.log_interval == 0:
                avg_loss = total_loss / num_batches
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch: {epoch:03d}/{cfg.training.epochs} | "
                    f"Batch: {batch_idx:04d}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )
                writer.add_scalar("train/loss", avg_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar("train/lr", current_lr, epoch * len(train_loader) + batch_idx)
        
        scheduler.step()
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        
        # Validation and sampling
        if rank == 0:
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if cfg.distributed else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }
            save_checkpoint(checkpoint, checkpoint_path)
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_checkpoint_path = Path(cfg.logging.checkpoint_dir) / "best_checkpoint.pth"
                save_checkpoint(checkpoint, best_checkpoint_path)
            
            # Sample some images periodically
            if epoch % cfg.logging.sample_interval == 0 or epoch == cfg.training.epochs - 1:
                model.eval()
                
                with torch.no_grad():
                    samples = ddpm.sample(
                        num_samples=cfg.logging.num_samples,
                        img_size=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)
                    )
                
                # Save samples
                save_path = Path(cfg.logging.sample_dir) / f"samples_epoch_{epoch:03d}.png"
                torchvision.utils.save_image(samples, save_path, nrow=int(np.sqrt(cfg.logging.num_samples)))
                
                # Log images to tensorboard
                grid = torchvision.utils.make_grid(samples, nrow=int(np.sqrt(cfg.logging.num_samples)))
                writer.add_image("generated_samples", grid, epoch)
            
            # Log epoch metrics
            writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
            writer.add_scalar("epoch/lr", optimizer.param_groups[0]['lr'], epoch)
    
    if rank == 0:
        writer.close()
        logging.info("Training completed!")

if __name__ == "__main__":
    main()