import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import os
import hydra
import yaml
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
from data.test_flower_dataset import FlowerDataset

def setup_logging(cfg, rank):
    """Setup logging and tensorboard writer"""
    if rank != 0:
        return None
    
    # Create directories if they don't exist
    Path(cfg.logging.tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.sample_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    def get_flower_loaders(batch_size, image_size, distributed=False):
        """Create train and validation dataloaders for flower dataset"""
        train_dataset = FlowerDataset(
            root_dir=cfg.data.root_dir,
            image_size=image_size,
            mode='train'
        )
        val_dataset = FlowerDataset(
            root_dir=cfg.data.root_dir,
            image_size=image_size,
            mode='val'
        )
        
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_dataset)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, train_sampler
        
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
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay
    )
    
    # Data loaders
    train_loader, val_loader, train_sampler = get_flower_loaders(
        batch_size=cfg.training.batch_size,
        image_size=cfg.data.image_size,
        distributed=cfg.distributed
    )
    
    # Training loop
    for epoch in range(cfg.training.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(rank)
            
            loss = ddpm.train_step(images, optimizer)  # Now returns a tensor
            
            # Backpropagate and update weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()  # Now safe to call .item()
            
            if rank == 0 and batch_idx % cfg.logging.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch: {epoch:03d}/{cfg.training.epochs} | "
                    f"Batch: {batch_idx:04d}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                )
                writer.add_scalar("train/loss", avg_loss, epoch * len(train_loader) + batch_idx)
        
        # Validation and sampling
        if rank == 0:
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
            epoch_loss = total_loss / len(train_loader)
            writer.add_scalar("epoch/train_loss", epoch_loss, epoch)
    
    if rank == 0:
        # Create save directory with cfg.logging.save_name
        save_dir = Path(cfg.logging.save_dir) / cfg.logging.save_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        model_save_path = save_dir / 'model.pth'
        state_dict = model.state_dict()
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Save only the state_dict
        torch.save(new_state_dict, model_save_path)
        logging.info(f'Model saved to {model_save_path}')

        # Save config by copying the original config file
        import shutil
        original_config_path = Path(hydra.utils.get_original_cwd()) / "config" / "base.yaml"
        config_save_path = save_dir / 'config.yaml'
        shutil.copy(original_config_path, config_save_path)
        logging.info(f'Config copied to {config_save_path}')

        writer.close()
        logging.info("Training completed!")

if __name__ == "__main__":
    main()