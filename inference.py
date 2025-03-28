import torch
import torchvision
import numpy as np
from pathlib import Path
import argparse
from models.unet import UNet
from ame_furi.ddpm import DDPM, LinearNoiseScheduler
import hydra
from omegaconf import OmegaConf

def load_model(config_path, device="cuda"):
    """Load trained model from checkpoint using config"""
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Initialize model
    model = UNet(
        n_channels=cfg.model.in_channels,
        n_classes=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        depth=cfg.model.depth,
        time_emb_dim=cfg.model.time_emb_dim
    ).to(device)
    
    # Load checkpoint (path now comes from config)
    checkpoint = torch.load(Path(cfg.logging.save_dir) / cfg.logging.save_name / 'model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize DDPM
    noise_scheduler = LinearNoiseScheduler(
        num_timesteps=cfg.diffusion.num_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end
    )
    
    ddpm = DDPM(model, noise_scheduler, device=device)
    return ddpm, cfg

def generate_images(ddpm, cfg):
    """Generate sample images using parameters from config"""
    output_dir = cfg.inference.output_dir
    num_samples = cfg.inference.num_samples
    
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        samples = ddpm.sample(
            num_samples=num_samples,
            img_size=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)
        )
    
    # Save samples
    for i in range(num_samples):
        save_path = Path(output_dir) / f"generated_flower_{i}.png"
        torchvision.utils.save_image(samples[i], save_path)
    
    return samples

def main():
    parser = argparse.ArgumentParser(description="Generate flower images using trained DDPM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    ddpm, cfg = load_model(args.config, device)
    
    # Generate images
    print(f"Generating {cfg.inference.num_samples} flower images...")
    samples = generate_images(ddpm, cfg)
    
    print(f"Images saved to {cfg.inference.output_dir}")

if __name__ == "__main__":
    main()