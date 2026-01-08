# src/test_pipeline.py

import torch

from data.dataset_word import WordDataset
from data.transform import make_log_mel_transform
from models.word_classifier import WordClassifier


def main():
    # 1) Skapa transform (log-mel-spektrogram)
    transform = make_log_mel_transform(
        sample_rate=16_000,
        n_mels=64,
    )
    print("Transform skapad.")

    # 2) Skapa dataset
    dataset = WordDataset(
        transform=transform,
        data_dirs=[
            "data/wav/sju",
            "data/wav/korsord",
            "data/wav/kraftskiva",
        ],
        target_sample_rate=16_000,
    )
    print("Dataset skapat. Antal filer:", len(dataset))

    if len(dataset) == 0:
        print("Inga wav-filer hittades. Kolla sökvägarna.")
        return

    # 3) Testa ett exempel
    features, label = dataset[0]
    print("Features shape:", features.shape)  # förväntat [1, n_mels, time]
    print("Label index:", label.item())

    # 4) Skapa modell
    n_mels = features.shape[1]
    model = WordClassifier(n_mels=n_mels, n_classes=3)
    print("Modell skapad.")

    # 5) Gör en dummy-batch (lägg på batch-dimension)
    x_batch = features.unsqueeze(0)  # [1, 1, n_mels, time]
    logits = model(x_batch)

    print("Logits shape:", logits.shape)  # förväntat [1, 3]
    print("Logits:", logits)


if __name__ == "__main__":
    main()