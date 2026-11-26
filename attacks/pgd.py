"""
Projected Gradient Descent (PGD) attack.
"""

import torch
import torch.nn as nn

def pgd_attack(model, x, y, epsilon=0.05, alpha=0.01, steps=10):
    """Generates adversarial examples using PGD."""
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * torch.sign(grad)
        # Project
        x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
    return x_adv
