import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from models.unet import UNet

class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1. - self.alpha_cumprod)

    def add_noise(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        t = t.to(self.sqrt_alpha_cumprod.device).long()
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
        return x_t, noise
    
    def sample_prev_timestep(self, x_t, model_output, t):
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = torch.sqrt(1.0 / (1 - beta_t))

        x_prev = sqrt_recip_alpha_t * (x_t - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            posterior_variance = self.betas[t] * (1. - self.alpha_cumprod[t-1]) / (1. - self.alpha_cumprod[t])
            x_prev += torch.sqrt(posterior_variance.view(-1, 1, 1, 1)) * noise

        return x_prev

class DDPM:
    def __init__(self, model, noise_scheduler, device="cuda"):
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.loss_fn = nn.MSELoss()

    def train_step(self, x_0, class_labels, optimizer):
        self.model.train()
        optimizer.zero_grad()

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device).long()

        noise = torch.randn_like(x_0)
        x_t, true_noise = self.noise_scheduler.add_noise(x_0, t, noise)

        # Pass class_labels to the model
        pred_noise = self.model(x_t, t, class_labels)

        loss = self.loss_fn(pred_noise, true_noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=1, img_size=(3, 64, 64), class_labels=None, progress=True):
        self.model.eval()
        x_t = torch.randn((num_samples, *img_size), device=self.device)
        
        # If no class labels provided, generate random ones
        if class_labels is None:
            class_labels = torch.randint(0, self.model.num_classes, (num_samples,), device=self.device)
        else:
            # Ensure class_labels is a tensor with correct shape
            class_labels = torch.tensor(class_labels, device=self.device).long()
            if len(class_labels.shape) == 0:
                class_labels = class_labels.unsqueeze(0)
            class_labels = class_labels.repeat(num_samples // len(class_labels) + 1)[:num_samples]

        iterator = reversed(range(self.noise_scheduler.num_timesteps))
        if progress:
            iterator = tqdm(iterator, desc="Sampling", total=self.noise_scheduler.num_timesteps)

        for t in iterator:
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x_t, t_batch, class_labels)
            x_t = self.noise_scheduler.sample_prev_timestep(x_t, pred_noise, t_batch)

        x_t = torch.clamp(x_t, -1., 1.)
        x_t = (x_t + 1.) / 2.
        return x_t