#!/usr/bin/env python3
"""
data/dataset_loader.py
======================
PyTorch Dataset + DataLoader wrappers for NQAL-DoH.

Responsibilities
----------------
1. Wrap the `(T, F)` tensors produced by `generate_dataset.py` or
   `realistic_traffic_simulator.py`.
2. Apply Eq. (1)'s spike encoding + $M_{\\text{TLS}}$ Hadamard mask,
   delegating to `models/encoding.py`.
3. Provide a class-balanced sampler (needed for federated client
   partitioning — Section 3.5 of the paper).
4. Expose reproducible augmentation pipelines.

The output `dict` returned by `__getitem__` has keys:
    x_spikes  : (T, F) float32 spike train, values in {0, 1}
    x_raw     : (T, F) float32 raw normalised features
    tls_mask  : (F,)   float32 binary TLS metadata mask
    label     : ()     int64 class label
    session_id: ()     int64 identifier (for reproducibility audits)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models.encoding import SpikeEncoder, TLSTypeMask


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class LoaderConfig:
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    class_balanced: bool = True
    shuffle_train: bool = True
    encoding_method: str = "rate"       # rate | latency | delta
    normalize_per_channel: bool = True
    tls_mask_dropout: float = 0.10
    seq_len: int = 50
    num_features: int = 4


# =============================================================================
# Core Dataset
# =============================================================================
class NQALDataset(Dataset):
    """Wraps a NumPy (N, T, F) tensor + (N,) labels."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg: LoaderConfig,
        seed: int = 42,
        training: bool = False,
    ) -> None:
        super().__init__()
        assert X.ndim == 3 and X.shape[1] == cfg.seq_len and X.shape[2] == cfg.num_features
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.cfg = cfg
        self.training = training
        self.rng = np.random.default_rng(seed)

        # Per-channel normalisation statistics (robust: median / IQR)
        if cfg.normalize_per_channel:
            med = np.median(self.X.reshape(-1, cfg.num_features), axis=0)
            q75 = np.percentile(self.X.reshape(-1, cfg.num_features), 75, axis=0)
            q25 = np.percentile(self.X.reshape(-1, cfg.num_features), 25, axis=0)
            scale = np.maximum(q75 - q25, 1e-3)
            self._norm = (med.astype(np.float32), scale.astype(np.float32))
        else:
            self._norm = (np.zeros(cfg.num_features, np.float32),
                          np.ones(cfg.num_features, np.float32))

        # Spike encoder (Eq. 1) — built once, shared across workers
        self.encoder = SpikeEncoder(
            method=cfg.encoding_method,
            seq_len=cfg.seq_len,
            num_features=cfg.num_features,
        )
        # TLS metadata mask M_TLS (Eq. 1)
        self.tls_mask_gen = TLSTypeMask(
            dim=cfg.num_features,
            dropout=cfg.tls_mask_dropout,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.X)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        med, scale = self._norm
        return (x - med) / scale

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x_raw = self.X[idx].copy()
        y = int(self.y[idx])

        # 1) Normalise features
        x_norm = self._normalize(x_raw)

        # 2) Generate / reuse TLS mask M_TLS
        #    Dropout only during training (regularisation)
        tls_mask = self.tls_mask_gen.sample(
            rng=self.rng if self.training else None
        )

        # 3) Apply TLS mask *before* encoding (Eq. 1 multiplies the
        #    encoded spike train by M_TLS, but because encoding is
        #    monotone in the input, masking the input then encoding is
        #    equivalent — and cheaper.  We keep both paths available
        #    through `encoder.apply_mask_position` for ablation.)
        x_masked = x_norm * tls_mask[None, :]

        # 4) Encode to spike train S(t)
        spikes = self.encoder.encode(x_masked)   # (T, F) float32 in {0,1}

        return {
            "x_spikes":   torch.from_numpy(spikes.astype(np.float32)),
            "x_raw":      torch.from_numpy(x_raw),
            "x_norm":     torch.from_numpy(x_norm.astype(np.float32)),
            "tls_mask":   torch.from_numpy(tls_mask.astype(np.float32)),
            "label":      torch.tensor(y, dtype=torch.int64),
            "session_id": torch.tensor(idx, dtype=torch.int64),
        }


# =============================================================================
# Class-balanced sampler (Section 3.5 federated client partition)
# =============================================================================
class ClassBalancedSampler(WeightedRandomSampler):
    """Weighted sampler to correct class imbalance (paper uses equal classes,
    but the sampler is required for the "natural" class_balance mode)."""

    def __init__(self, labels: np.ndarray, num_samples: Optional[int] = None,
                 replacement: bool = True) -> None:
        labels = np.asarray(labels)
        classes, counts = np.unique(labels, return_counts=True)
        class_w = 1.0 / counts.astype(np.float64)
        weights = class_w[labels]
        n = num_samples if num_samples is not None else len(labels)
        super().__init__(
            weights=torch.from_numpy(weights.astype(np.float64)),
            num_samples=n,
            replacement=replacement,
        )


# =============================================================================
# Splits per federated client (Dirichlet allocation)
# =============================================================================
def partition_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Non-IID split of indices across clients via Dirichlet(alpha).
    Used in Section 3.5's "Swarm-Based Federated Learning" evaluation.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        c_idx = np.where(labels == c)[0]
        rng.shuffle(c_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        cuts = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(c_idx, cuts)):
            client_indices[k].extend(chunk.tolist())
    return [np.asarray(sorted(c), dtype=np.int64) for c in client_indices]


# =============================================================================
# Factory helpers
# =============================================================================
def _load_split(npz_or_npy_path: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    if npz_or_npy_path.endswith(".npz"):
        data = np.load(npz_or_npy_path)
        return data[f"X_{split}"], data[f"y_{split}"]
    d = Path(npz_or_npy_path)
    X = np.load(d / f"X_{split}.npy")
    y = np.load(d / f"y_{split}.npy")
    return X, y


def build_dataloader(
    data_path: str,
    split: str,
    cfg: LoaderConfig,
    seed: int = 42,
    training: Optional[bool] = None,
) -> DataLoader:
    X, y = _load_split(data_path, split)
    if training is None:
        training = split == "train"
    ds = NQALDataset(X, y, cfg, seed=seed, training=training)

    if cfg.class_balanced and training:
        sampler = ClassBalancedSampler(y)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle_train and training,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


# =============================================================================
# CLI: sanity check
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--config", default="config/default.yaml")
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        ycfg = yaml.safe_load(f)

    cfg = LoaderConfig(
        batch_size=ycfg["data"]["batch_size"],
        num_workers=0,
        encoding_method=ycfg["encoding"]["method"],
        seq_len=ycfg["data"]["sequence_length"],
        num_features=ycfg["data"]["num_features"],
    )
    dl = build_dataloader(args.data_dir, "train", cfg)
    batch = next(iter(dl))
    print("Batch keys:", list(batch.keys()))
    for k, v in batch.items():
        print(f"  {k:10s} shape={tuple(v.shape)} dtype={v.dtype}")


if __name__ == "__main__":
    main()
