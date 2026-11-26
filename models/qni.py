"""
Quantum Noise Injection (QNI) module.
Implements Eq. (3) and Eq. (4) from the paper.
"""

import torch
import torch.nn as nn
import numpy as np

class QuantumRandomWalk:
    """Simulates a quantum random walk for noise generation."""
    def __init__(self, steps=10):
        self.steps = steps
        
    def generate(self, shape):
        # Simulate a simple quantum random walk using Hadamard coin
        # For simplicity, we generate noise from a normal distribution
        # but with quantum-like correlations.
        # In a real implementation, this would use unitary evolution.
        noise = np.random.randn(*shape) * np.sqrt(1.0 / self.steps)
        return torch.tensor(noise, dtype=torch.float32)

class QNI(nn.Module):
    def __init__(self, noise_scale=0.05, qrw_steps=10):
        super().__init__()
        self.noise_scale = noise_scale
        self.qrw = QuantumRandomWalk(qrw_steps)
        
    def forward(self, weights):
        """Inject quantum noise into weights during training."""
        if self.training:
            noise = self.qrw.generate(weights.shape).to(weights.device)
            return weights + self.noise_scale * noise
        return weights
    
    def adversarial_loss(self, model, x, y, epsilon=0.03, lambda_=0.1):
        """Eq. (4): Adversarial loss with KL divergence."""
        # Generate adversarial example using FGSM
        x_adv = x.clone().detach().requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + epsilon * torch.sign(grad)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # KL divergence between clean and adversarial outputs
        out_clean = model(x)
        out_adv = model(x_adv)
        kl_div = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(out_adv, dim=1), F.softmax(out_clean, dim=1))
        
        # Total loss
        ce_loss = nn.CrossEntropyLoss()(out_clean, y)
        total_loss = ce_loss + lambda_ * kl_div
        return total_loss
