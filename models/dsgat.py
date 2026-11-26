"""
Dynamic Spiking Graph Attention (DSGAT) module.
Implements Eq. (2) from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DSGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.theta_q = nn.Linear(in_dim, out_dim * num_heads)
        self.theta_k = nn.Linear(in_dim, out_dim * num_heads)
        self.theta_v = nn.Linear(in_dim, out_dim * num_heads)
        self.out_proj = nn.Linear(out_dim * num_heads, out_dim)
        
    def forward(self, x, adj=None):
        # x: (batch, time, in_dim) spike trains
        batch, time, _ = x.shape
        
        # Reshape to (batch, time, num_heads, out_dim)
        Q = self.theta_q(x).view(batch, time, self.num_heads, self.out_dim)
        K = self.theta_k(x).view(batch, time, self.num_heads, self.out_dim)
        V = self.theta_v(x).view(batch, time, self.num_heads, self.out_dim)
        
        # Attention scores: (batch, num_heads, time, time)
        scores = torch.einsum('bthd,bshd->bhts', Q, K) / math.sqrt(self.out_dim)
        
        # Apply mask if adjacency provided (optional)
        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(1) == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Aggregate
        out = torch.einsum('bhts,bshd->bthd', attn, V)
        out = out.reshape(batch, time, self.num_heads * self.out_dim)
        out = self.out_proj(out)
        return out

class DSGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.layer1 = DSGATLayer(in_dim, hidden_dim, num_heads, dropout)
        self.layer2 = DSGATLayer(hidden_dim, hidden_dim, num_heads, dropout)
        
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x
