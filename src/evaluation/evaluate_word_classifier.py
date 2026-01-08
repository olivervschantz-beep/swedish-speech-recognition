# src/evaluation/evaluate_word_classifier.py

import argparse
import os
import torch
import torch.nn.functional as F  # <--- TILLAGD IMPORT
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Importera projekt-specifika moduler
from src.data.dataset_word import WordDataset
from src.data.transform import make_log_mel_transform
from src.models.word_classifier import WordClassifier


def get_device() -> torch.device:
    """Välj GPU om möjligt, annars CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Mac med Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# vvvvvv TILLAGD FUNKTION vvvvvv
def pad_collate_fn(batch):
    """
    En anpassad collate_fn som paddar spektrogram i en batch 
    till samma längd (tidsdimension).
    """
    # batch är en lista av (features, label)
    
    # 1. Separera features och labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 2. Hitta maxlängden i tidsdimensionen (dim 2)
    max_len = max([f.shape[2] for f in features])

    # 3. Padda alla features till max_len
    padded_features = []
    for f in features:
        pad_width = max_len - f.shape[2]
        padded_f = F.pad(f, (0, pad_width), "constant", 0)
        padded_features.append(padded_f)

    # 4. Stacka de lika stora tensorerna
    features_batch = torch.stack(padded_features)
    
    # 5. Stacka etiketterna
    labels_batch = torch.stack(labels)

    return features_batch, labels_batch
# ^^^^^^ SLUT PÅ TILLAGD FUNKTION ^^^^^^


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained WordClassifier model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/word_classifier.pt",
        help="Sökväg till den tränade modellfilen (.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/test", # Använder test-mappen som standard
        help="Rotmapp som innehåller SJU/, KORSORD/, KRAFTSKIVA/ att evaluera på.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch-storlek för evaluering."
    )

    args = parser.parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # 1. Ladda Checkpoint (modell + metadata)
    if not os.path.exists(args.model):
        print(f"Modellfil hittades inte: {args.model}")
        return

    checkpoint = torch.load(args.model, map_location=device)

    # Hämta metadata från checkpointen
    n_mels = checkpoint.get('n_mels')
    idx_to_label = checkpoint.get('idx_to_label')
    
    if n_mels is None or idx_to_label is None:
        print("Checkpoint-filen saknar 'n_mels' eller 'idx_to_label'.")
        return
        
    # Konvertera sträng-nycklar (från sparande) till int-nycklar
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    n_classes = len(idx_to_label)
    label_names = [idx_to_label[i] for i in range(n_classes)]
    
    print(f"Laddar modell med n_mels={n_mels}, n_classes={n_classes}")
    print(f"Etiketter: {label_names}")

    # 2. Initiera Modell och ladda vikter
    model = WordClassifier(n_mels=n_mels, n_classes=n_classes).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()  

    # 3. Förbered Transform och Dataloader
    transform = make_log_mel_transform(
        sample_rate=16_000,
        n_mels=n_mels,
    )

    # Bygg sökvägarna till datamapparna (med gemener)
    data_dirs = [os.path.join(args.data_root, name.lower()) for name in label_names]

    test_dataset = WordDataset(
        transform=transform,
        data_dirs=data_dirs,
        target_sample_rate=16_000,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn  # <--- TILLAGD RAD
    )

    # 4. Evalueringsloop
    y_true = []
    y_pred = []

    print(f"Startar evaluering på {len(test_dataset)} samplingar...")
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 5. Beräkna och skriv ut resultat
    print("\n" + "="*30)
    print("      ORDKLASSIFICERINGSRESULTAT      ")
    print("="*30 + "\n")

    # Total Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Total Accuracy: {accuracy * 100:.2f}%\n")

    # Klassificeringsrapport
    report = classification_report(y_true, y_pred, target_names=label_names)
    print("Klassificeringsrapport:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("(Rad = Sann etikett, Kolumn = Predikterad etikett)\n")
    
    # Snyggare utskrift av CM
    header = "Pred: " + " | ".join([f"{name[:10]}" for name in label_names])
    print(f"{'':<14} | {header}")
    print("-" * (len(header) + 16))
    for i, row in enumerate(cm):
        row_str = " | ".join([f"{val:10d}" for val in row])
        print(f"Sann: {label_names[i]:<10} | {row_str}")


if __name__ == "__main__":
    # Kör från projektroten:
    #   python -m src.evaluation.evaluate_word_classifier --data-root data/test
    main()