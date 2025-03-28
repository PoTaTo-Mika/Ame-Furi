import torch
import torchvision
import numpy as np
from pathlib import Path
import argparse
import logging
from models.unet import UNet
from ame_furi.ddpm import DDPM, LinearNoiseScheduler
import hydra
from omegaconf import OmegaConf

def setup_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_model(config_path, device="cuda"):
    """Load trained model from checkpoint using config"""
    try:
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
        
        # Construct checkpoint path
        checkpoint_path = Path(cfg.logging.save_dir) / cfg.logging.save_name / 'model.pth'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        # Load checkpoint (only state_dict is stored)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        # Initialize DDPM
        noise_scheduler = LinearNoiseScheduler(
            num_timesteps=cfg.diffusion.num_timesteps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end
        )
        
        ddpm = DDPM(model, noise_scheduler, device=device)
        return ddpm, cfg
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def generate_images(ddpm, cfg, device="cuda"):
    """Generate sample images using parameters from config"""
    try:
        output_dir = Path(cfg.inference.output_dir)
        num_samples = cfg.inference.num_samples
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        logging.info(f"Generating {num_samples} samples...")
        with torch.no_grad():
            samples = ddpm.sample(
                num_samples=num_samples,
                img_size=(cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)
            )
        
        # Save samples
        for i in range(num_samples):
            save_path = output_dir / f"generated_flower_{i}.png"
            torchvision.utils.save_image(samples[i], save_path)
            logging.info(f"Saved image to {save_path}")
        
        return samples
        
    except Exception as e:
        logging.error(f"Error generating images: {str(e)}")
        raise

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate flower images using trained DDPM")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    try:
        # Load model
        ddpm, cfg = load_model(args.config, device)
        
        # Generate images
        samples = generate_images(ddpm, cfg, device)
        
        logging.info(f"Successfully generated {cfg.inference.num_samples} images")
        logging.info(f"Images saved to {cfg.inference.output_dir}")
        
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()