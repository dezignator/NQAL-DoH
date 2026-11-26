"""
Spiking Neural Network with Leaky Integrate-and-Fire (LIF) neurons.
Implements surrogate gradient for backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, v_threshold):
        ctx.save_for_backward(input)
        ctx.v_threshold = v_threshold
        return (input >= v_threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        v_threshold = ctx.v_threshold
        # Surrogate gradient: fast sigmoid
        grad_input = grad_output * (1.0 / (1.0 + torch.abs(input - v_threshold) * 10.0) ** 2)
        return grad_input, None

class LIFLayer(nn.Module):
    def __init__(self, input_size, hidden_size, tau_mem=20.0, tau_syn=5.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        
        self.linear = nn.Linear(input_size, hidden_size)
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('syn', torch.zeros(1, hidden_size))
    
    def reset(self, batch_size):
        self.mem = torch.zeros(batch_size, self.hidden_size, device=self.mem.device)
        self.syn = torch.zeros(batch_size, self.hidden_size, device=self.syn.device)
    
    def forward(self, x):
        # x: (batch, time, input_size)
        batch_size, time_steps, _ = x.shape
        self.reset(batch_size)
        spikes = []
        for t in range(time_steps):
            input_current = self.linear(x[:, t, :])
            # Synaptic update
            self.syn = self.syn + (-self.syn / self.tau_syn) + input_current
            # Membrane update
            self.mem = self.mem + (-self.mem / self.tau_mem) + self.syn
            # Spike
            spike = SurrogateSpike.apply(self.mem, self.v_threshold)
            # Reset
            self.mem = self.mem * (1 - spike) + self.v_reset * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1)  # (batch, time, hidden_size)

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, tau_mem=20.0, tau_syn=5.0, v_threshold=1.0):
        super().__init__()
        self.lif1 = LIFLayer(input_size, hidden_size, tau_mem, tau_syn, v_threshold)
        self.lif2 = LIFLayer(hidden_size, hidden_size, tau_mem, tau_syn, v_threshold)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, time, input_size)
        spikes1 = self.lif1(x)
        spikes2 = self.lif2(spikes1)
        # Aggregate over time (mean firing rate)
        out = spikes2.mean(dim=1)
        return self.fc(out)
