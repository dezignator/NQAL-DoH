"""
Evaluation script for NQAL.
Computes accuracy, FPR, detection rates, robustness, latency, and energy.
"""

import torch
import numpy as np
import yaml
import time
from models.snn import SNN
from models.dsgat import DSGAT
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.quantum_noise import quantum_noise_attack
from utils.metrics import compute_accuracy, compute_fpr, compute_detection_rate
from utils.energy import compute_energy

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    test_data = np.load('data/test.npz')
    X_test = torch.tensor(test_data['X'], dtype=torch.float32).to(device)
    y_test = torch.tensor(test_data['y'], dtype=torch.long).to(device)
    
    # Load model
    input_size = config['data']['num_features']
    hidden_size = config['model']['snn']['num_neurons']
    num_classes = config['data']['num_classes']
    snn = SNN(input_size, hidden_size, num_classes).to(device)
    snn.load_state_dict(torch.load('models/nqal_snn.pth', map_location=device))
    snn.eval()
    
    # Clean evaluation
    with torch.no_grad():
        output = snn(X_test)
        pred = output.argmax(dim=1)
        acc = compute_accuracy(y_test.cpu().numpy(), pred.cpu().numpy())
        fpr = compute_fpr(y_test.cpu().numpy(), pred.cpu().numpy())
    
    print(f"Clean Test Accuracy: {acc:.4f}, FPR: {fpr:.4f}")
    
    # Adversarial robustness
    results = {}
    for attack_name in config['evaluation']['attack_scenarios']:
        if attack_name == 'FGSM':
            X_adv = fgsm_attack(snn, X_test, y_test, epsilon=config['training']['fgsm_epsilon'])
        elif attack_name == 'PGD':
            X_adv = pgd_attack(snn, X_test, y_test, epsilon=config['training']['pgd_epsilon'], steps=config['training']['pgd_steps'])
        elif attack_name == 'QuantumNoise':
            X_adv = quantum_noise_attack(X_test, epsilon=config['model']['qni']['noise_scale'])
        else:
            continue
        
        with torch.no_grad():
            output_adv = snn(X_adv)
            pred_adv = output_adv.argmax(dim=1)
            acc_adv = compute_accuracy(y_test.cpu().numpy(), pred_adv.cpu().numpy())
        results[attack_name] = acc_adv
        print(f"{attack_name} Accuracy: {acc_adv:.4f}")
    
    # Detection rates per attack class (Table 2)
    print("\nDetection Rates (Table 2):")
    class_names = ['Benign', 'DNS Tunneling', 'C2 Beaconing', 'DGA', 'Adversarial DoH']
    with torch.no_grad():
        pred = snn(X_test).argmax(dim=1)
    for i, name in enumerate(class_names):
        if i == 0:
            continue
        dr = compute_detection_rate(y_test.cpu().numpy(), pred.cpu().numpy(), i)
        print(f"{name}: {dr:.4f}")
    
    # Latency measurement
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = snn(X_test[:1])
    end = time.time()
    latency_ms = (end - start) / 100 * 1000
    # Scale to hardware (simulated)
    hardware_latency_ms = latency_ms * config['hardware']['latency_scaling_factor']
    print(f"\nLatency (software): {latency_ms:.4f} ms, Simulated hardware: {hardware_latency_ms:.4f} ms")
    
    # Energy estimation
    # Count spikes (simplified: use number of neurons * time steps)
    spike_counts = [hidden_size * config['data']['sequence_length']]  # per inference
    weights = snn.fc.weight
    energy = compute_energy(spike_counts, weights, 
                            energy_per_spike_pj=config['hardware']['energy_per_spike_pj'],
                            sparsity_penalty_lambda=config['hardware']['sparsity_penalty_lambda'])
    print(f"Energy per inference: {energy:.4f} pJ")
    print(f"GPU baseline energy: {config['hardware']['gpu_energy_per_inference_mj']} mJ")
    print(f"Energy efficiency gain: {config['hardware']['gpu_energy_per_inference_mj'] * 1e9 / energy:.2f}x")

if __name__ == '__main__':
    main()
