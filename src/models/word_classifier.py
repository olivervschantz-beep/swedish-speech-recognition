# src/training/train_word_classifier.py
import argparse
import os
import torch
import torch.nn as nn


class WordClassifier(nn.Module):
    """
    Enkel CNN-modell för ordigenkänning.

    Indata:
        x: [batch, 1, n_mels, time] (log-mel-spektrogram)

    Utdata:
        logits: [batch, 3]
            0 = SJU
            1 = KORSORD
            2 = KRAFTSKIVA
    """

    def __init__(self, n_mels: int, n_classes: int = 3) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)      # [B, 32, H', W']
        x = self.global_pool(x)     # [B, 32, 1, 1]
        x = x.view(x.size(0), -1)   # [B, 32]
        logits = self.fc(x)         # [B, 3]
        return logits


# if __name__ == "__main__":
    # Liten självtest
    # batch_size = 4
    # n_mels = 64
    # time_steps = 100

    # model = WordClassifier(n_mels=n_mels, n_classes=3)
    # dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    # out = model(dummy_input)

    # print("Input shape :", dummy_input.shape)
    # print("Output shape:", out.shape)   # bör bli [4, 3]