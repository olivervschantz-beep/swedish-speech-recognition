# src/training/utils.py

"""
Gemensamma tränings-hjälpfunktioner:

- get_device: välj GPU/MPS/CPU
- train_one_epoch: kör en tränings-epoch över en DataLoader
- evaluate: evaluera modell på en DataLoader (loss + accuracy)
- set_seed: (valfri) gör experiment mer reproducerbara
"""

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch import nn


def get_device() -> torch.device:
    """
    Välj bästa tillgängliga device:
    - CUDA (Nvidia GPU) om möjligt
    - MPS (Apple Silicon) om möjligt
    - annars CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Mac med Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    Sätt random seed för mer reproducerbara resultat.
    Helt frivillig att använda.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Träna modellen i EN epoch över dataloadern.

    Returnerar:
    - avg_loss: viktat med batch-storlek (sum loss / antal samples)
    - avg_acc: andel korrekta prediktioner (0..1)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Utvärdera modellen på en DataLoader (ingen träning, bara framåtpass).

    Returnerar:
    - avg_loss
    - avg_acc
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc
