# Results

This directory contains the output of the NQAL evaluation.

## Expected Results (reproduced from paper)

| Metric | Value |
|--------|-------|
| Accuracy | 99.18% |
| False Positive Rate | < 0.5% |
| Latency | < 1 ms |
| Energy Efficiency | 10× better than GPU |
| Adversarial Robustness (FGSM/PGD) | 94% |
| QNI Exploit Prevention | 100% |

## Detection Rates per Attack (Table 2)

| Attack Type | Detection Rate |
|-------------|----------------|
| DNS Tunneling | 98.7% |
| C2 Beaconing | 99.2% |
| Adversarial DoH | 94.0% |
| DGA over DoH | 97.5% |
| QNI Exploits | 100% |

## Files

After running `run_experiments.sh`, the following files will be generated:
- `results/metrics.csv` – All evaluation metrics
- `results/table2.csv` – Detection rates per attack
- `results/energy_report.txt` – Energy consumption details
