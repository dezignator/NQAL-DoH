"""
Fast Gradient Sign Method (FGSM) attack.
"""

import torch
import torch.nn as nn

def fgsm_attack(model, x, y, epsilon=0.03):
    """Generates adversarial examples using FGSM."""
    x_adv = x.clone().detach().requires_grad_(True)
    output = model(x_adv)
    loss = nn.CrossEntropyLoss()(output, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * torch.sign(grad)
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()
