"""
Synthetic DoH traffic dataset generator.
Simulates benign and malicious patterns: DNS Tunneling, C2 Beaconing, DGA, Adversarial DoH.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_benign(num_samples, seq_len, num_features):
    """Benign DoH traffic: random packet sizes and inter-arrival times."""
    data = np.random.normal(loc=500, scale=100, size=(num_samples, seq_len, num_features))
    # Normalize
    data = np.clip(data, 0, None)
    return data

def generate_dns_tunneling(num_samples, seq_len, num_features):
    """DNS tunneling: high entropy, periodic small packets."""
    data = np.zeros((num_samples, seq_len, num_features))
    for i in range(num_samples):
        # Small packet sizes, high entropy
        data[i, :, 0] = np.random.uniform(50, 100, seq_len)  # packet size
        data[i, :, 1] = np.random.exponential(0.1, seq_len)  # inter-arrival time
        data[i, :, 2] = np.random.uniform(0.8, 1.0, seq_len)  # entropy
        data[i, :, 3] = np.random.choice([0, 1], seq_len)      # direction
    return data

def generate_c2_beaconing(num_samples, seq_len, num_features):
    """C2 beaconing: periodic callbacks with regular intervals."""
    data = np.zeros((num_samples, seq_len, num_features))
    for i in range(num_samples):
        period = np.random.uniform(5, 15)  # seconds
        data[i, :, 0] = np.random.normal(200, 20, seq_len)
        data[i, :, 1] = np.full(seq_len, period) + np.random.normal(0, 0.1, seq_len)
        data[i, :, 2] = np.random.uniform(0.4, 0.6, seq_len)
        data[i, :, 3] = np.random.choice([0, 1], seq_len)
    return data

def generate_dga(num_samples, seq_len, num_features):
    """DGA over DoH: algorithmically generated domains."""
    data = np.zeros((num_samples, seq_len, num_features))
    for i in range(num_samples):
        data[i, :, 0] = np.random.normal(300, 50, seq_len)
        data[i, :, 1] = np.random.exponential(1.0, seq_len)
        data[i, :, 2] = np.random.uniform(0.6, 0.9, seq_len)
        data[i, :, 3] = np.random.choice([0, 1], seq_len)
    return data

def generate_adversarial(num_samples, seq_len, num_features, epsilon=0.03):
    """Adversarial DoH: benign-like traffic with FGSM perturbations."""
    data = generate_benign(num_samples, seq_len, num_features)
    # Add FGSM-like perturbation (random sign)
    perturbation = epsilon * np.sign(np.random.randn(*data.shape))
    data = data + perturbation
    return data

def create_dataset(config):
    num_samples = config['data']['num_samples']
    seq_len = config['data']['sequence_length']
    num_features = config['data']['num_features']
    num_classes = config['data']['num_classes']
    
    # Generate equal number per class
    samples_per_class = num_samples // num_classes
    
    X = []
    y = []
    
    # Class 0: Benign
    X.append(generate_benign(samples_per_class, seq_len, num_features))
    y.extend([0] * samples_per_class)
    
    # Class 1: DNS Tunneling
    X.append(generate_dns_tunneling(samples_per_class, seq_len, num_features))
    y.extend([1] * samples_per_class)
    
    # Class 2: C2 Beaconing
    X.append(generate_c2_beaconing(samples_per_class, seq_len, num_features))
    y.extend([2] * samples_per_class)
    
    # Class 3: DGA
    X.append(generate_dga(samples_per_class, seq_len, num_features))
    y.extend([3] * samples_per_class)
    
    # Class 4: Adversarial DoH
    X.append(generate_adversarial(samples_per_class, seq_len, num_features))
    y.extend([4] * samples_per_class)
    
    X = np.concatenate(X, axis=0)
    y = np.array(y)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    train_ratio = config['data']['train_split']
    val_ratio = config['data']['val_split']
    test_ratio = config['data']['test_split']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1-train_ratio), random_state=config['data']['random_seed'], stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=config['data']['random_seed'], stratify=y_temp)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == '__main__':
    config = load_config()
    np.random.seed(config['data']['random_seed'])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_dataset(config)
    
    # Save to disk
    os.makedirs('data', exist_ok=True)
    np.savez('data/train.npz', X=X_train, y=y_train)
    np.savez('data/val.npz', X=X_val, y=y_val)
    np.savez('data/test.npz', X=X_test, y=y_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print("Dataset generated and saved to data/")
