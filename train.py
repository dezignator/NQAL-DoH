"""
Main training script for NQAL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from models.snn import SNN
from models.dsgat import DSGAT
from models.qni import QNI
from models.zkp import ZKPLearn
from models.federated import SwarmFederatedLearning
from models.gan import ThreatImmunizationGAN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.quantum_noise import quantum_noise_attack
from utils.metrics import compute_accuracy
from tqdm import tqdm

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data = np.load('data/train.npz')
    val_data = np.load('data/val.npz')
    X_train, y_train = torch.tensor(train_data['X'], dtype=torch.float32), torch.tensor(train_data['y'], dtype=torch.long)
    X_val, y_val = torch.tensor(val_data['X'], dtype=torch.float32), torch.tensor(val_data['y'], dtype=torch.long)
    
    # Model
    input_size = config['data']['num_features']
    hidden_size = config['model']['snn']['num_neurons']
    num_classes = config['data']['num_classes']
    
    snn = SNN(input_size, hidden_size, num_classes).to(device)
    dsgat = DSGAT(hidden_size, config['model']['dsgat']['hidden_dim']).to(device)
    qni = QNI(noise_scale=config['model']['qni']['noise_scale'], qrw_steps=config['model']['qni']['qrw_steps'])
    zkp = ZKPLearn(threshold_tau=config['model']['zkp']['threshold_tau'])
    gan = ThreatImmunizationGAN(latent_dim=config['model']['gan']['latent_dim'],
                                gen_hidden=config['model']['gan']['gen_hidden'],
                                disc_hidden=config['model']['gan']['disc_hidden'],
                                adv_weight=config['model']['gan']['adv_weight'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(snn.parameters()) + list(dsgat.parameters()), lr=config['training']['learning_rate'])
    
    # Federated learning setup
    fed = SwarmFederatedLearning(
        num_clients=config['model']['federated']['num_clients'],
        communication_rounds=config['model']['federated']['communication_rounds'],
        local_epochs=config['model']['federated']['local_epochs'],
        swarm_beta=config['model']['federated']['swarm_beta']
    )
    
    # Split training data among clients
    client_data = []
    client_labels = []
    num_clients = config['model']['federated']['num_clients']
    indices = np.array_split(np.arange(len(X_train)), num_clients)
    for idx in indices:
        client_data.append(X_train[idx].to(device))
        client_labels.append(y_train[idx].to(device))
    
    # GAN training (pre-training)
    print("Training GAN for threat immunization...")
    optimizer_G = optim.Adam(gan.generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(gan.discriminator.parameters(), lr=0.0002)
    for epoch in range(10):
        for i in range(0, len(X_train), config['training']['batch_size']):
            real = X_train[i:i+config['training']['batch_size']].to(device)
            d_loss, g_loss = gan.train_step(real, optimizer_G, optimizer_D, criterion)
        print(f"GAN Epoch {epoch+1}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")
    
    # Federated training
    print("Starting federated training...")
    for round_idx in tqdm(range(fed.communication_rounds)):
        # Local training on each client
        client_weights = []
        for client_idx in range(num_clients):
            # Local model
            local_snn = SNN(input_size, hidden_size, num_classes).to(device)
            local_snn.load_state_dict(snn.state_dict())
            local_optimizer = optim.Adam(local_snn.parameters(), lr=config['training']['learning_rate'])
            
            local_snn.train()
            for epoch in range(fed.local_epochs):
                for i in range(0, len(client_data[client_idx]), config['training']['batch_size']):
                    x = client_data[client_idx][i:i+config['training']['batch_size']]
                    y = client_labels[client_idx][i:i+config['training']['batch_size']]
                    
                    local_optimizer.zero_grad()
                    # Apply QNI to weights
                    for name, param in local_snn.named_parameters():
                        param.data = qni(param.data)
                    
                    output = local_snn(x)
                    loss = criterion(output, y)
                    
                    # Add adversarial loss (Eq. 4)
                    if config['training']['adversarial_training']:
                        x_adv = fgsm_attack(local_snn, x, y, epsilon=config['training']['fgsm_epsilon'])
                        output_adv = local_snn(x_adv)
                        loss += 0.1 * criterion(output_adv, y)
                    
                    loss.backward()
                    local_optimizer.step()
            
            client_weights.append(local_snn.state_dict())
        
        # Aggregate using swarm rule (Eq. 7)
        global_weights = fed.aggregate(client_weights)
        snn.load_state_dict(global_weights)
        
        # Validate
        snn.eval()
        with torch.no_grad():
            val_output = snn(X_val.to(device))
            val_pred = val_output.argmax(dim=1)
            val_acc = compute_accuracy(y_val.cpu().numpy(), val_pred.cpu().numpy())
        print(f"Round {round_idx+1}/{fed.communication_rounds}, Val Acc: {val_acc:.4f}")
    
    # Save model
    torch.save(snn.state_dict(), 'models/nqal_snn.pth')
    print("Training complete. Model saved to models/nqal_snn.pth")

if __name__ == '__main__':
    main()
