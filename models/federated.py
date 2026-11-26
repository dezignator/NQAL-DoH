"""
Swarm-Based Federated Learning module.
Implements Eq. (7) from the paper.
"""

import torch
import torch.nn as nn
import copy
import numpy as np

class SwarmFederatedLearning:
    def __init__(self, num_clients=10, communication_rounds=20, local_epochs=3, swarm_beta=0.1):
        self.num_clients = num_clients
        self.communication_rounds = communication_rounds
        self.local_epochs = local_epochs
        self.swarm_beta = swarm_beta
        
    def pheromone_nonlinearity(self, x):
        """Eq. (7): phi function."""
        return torch.tanh(x)  # simple nonlinearity
    
    def aggregate(self, client_weights):
        """Eq. (7): Swarm update rule."""
        # client_weights: list of state_dicts
        global_weights = copy.deepcopy(client_weights[0])
        
        for key in global_weights:
            # Stack weights from all clients
            weights = torch.stack([w[key] for w in client_weights])
            # Compute pheromone-inspired nonlinearity
            phi_weights = self.pheromone_nonlinearity(weights)
            # Weighted average
            aggregated = phi_weights.mean(dim=0)
            # Add QNI noise (simulated)
            noise = torch.randn_like(aggregated) * self.swarm_beta
            global_weights[key] = aggregated + noise
            
        return global_weights
    
    def train(self, global_model, client_data, client_labels, criterion, optimizer_fn, epochs=1):
        """Federated training loop."""
        global_weights = global_model.state_dict()
        
        for round_idx in range(self.communication_rounds):
            client_weights = []
            
            for client_idx in range(self.num_clients):
                # Create local model
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(global_weights)
                local_optimizer = optimizer_fn(local_model.parameters())
                
                # Local training
                local_model.train()
                for epoch in range(self.local_epochs):
                    # Get client data (simplified: use all data for each client)
                    x = client_data[client_idx]
                    y = client_labels[client_idx]
                    
                    optimizer = local_optimizer
                    optimizer.zero_grad()
                    output = local_model(x)
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                
                client_weights.append(local_model.state_dict())
            
            # Aggregate
            global_weights = self.aggregate(client_weights)
        
        global_model.load_state_dict(global_weights)
        return global_model
