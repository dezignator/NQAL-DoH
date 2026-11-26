"""
Threat Immunization Engine using Adversarial GANs.
Implements Eq. (8) from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class ThreatImmunizationGAN:
    def __init__(self, latent_dim=32, gen_hidden=64, disc_hidden=64, adv_weight=0.1):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, gen_hidden)
        self.discriminator = Discriminator(gen_hidden, disc_hidden)
        self.adv_weight = adv_weight
        
    def train_step(self, real_data, optimizer_G, optimizer_D, criterion):
        """Eq. (8): Adversarial GAN objective."""
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, device=device)
        fake_data = self.generator(z)
        
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data.detach())
        
        d_loss = -torch.mean(torch.log(real_pred + 1e-8) + torch.log(1 - fake_pred + 1e-8))
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        fake_pred = self.discriminator(fake_data)
        # Feature matching loss (simplified)
        g_loss = -torch.mean(torch.log(fake_pred + 1e-8))
        # Add regularization
        g_loss += self.adv_weight * torch.norm(fake_data, p='fro')
        g_loss.backward()
        optimizer_G.step()
        
        return d_loss.item(), g_loss.item()
    
    def generate_adversarial_samples(self, num_samples, device='cpu'):
        """Generate synthetic malicious samples."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self.generator(z)
        return samples
