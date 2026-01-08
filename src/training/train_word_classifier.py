# src/training/train_word_classifier.py

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset # <--- Subset ÄR NY

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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
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

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def pad_collate_fn(batch):
    """
    En anpassad collate_fn som paddar spektrogram i en batch 
    till samma längd (tidsdimension).
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_len = max([f.shape[2] for f in features])

    padded_features = []
    for f in features:
        pad_width = max_len - f.shape[2]
        padded_f = F.pad(f, (0, pad_width), "constant", 0)
        padded_features.append(padded_f)

    features_batch = torch.stack(padded_features)
    labels_batch = torch.stack(labels)

    return features_batch, labels_batch


def main():
    parser = argparse.ArgumentParser(
        description="Train word classifier for SJU / KORSORD / KRAFTSKIVA."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/wav",
        help="Rotmapp som innehåller sju/, korsord/, kraftskiva/ (default: data/wav)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--output", type=str, default="models/word_classifier.pt")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # 1) Transform (features)
    transform = make_log_mel_transform(
        sample_rate=16_000,
        n_mels=args.n_mels,
    )

    # 2) Dataset
    data_dirs = [
        os.path.join(args.data_root, "sju"),
        os.path.join(args.data_root, "korsord"),
        os.path.join(args.data_root, "kraftskiva"),
    ]
    
    # --- STOR ÄNDRING HÄR ---
    # Vi skapar TVÅ dataset-instanser så att vi kan slå på/av förstärkning
    
    # Dataset för TRÄNING (MED förstärkning)
    train_dataset_instance = WordDataset(
        transform=transform,
        data_dirs=data_dirs,
        target_sample_rate=16_000,
        apply_augmentation=True  # <--- PÅ
    )
    
    # Dataset för VALIDERING (UTAN förstärkning)
    val_dataset_instance = WordDataset(
        transform=transform,
        data_dirs=data_dirs,
        target_sample_rate=16_000,
        apply_augmentation=False # <--- AV
    )

    # Manuell uppdelning av index
    # Vi måste se till att båda dataseten (train och val)
    # drar från samma pool av filer, men att val_ds inte
    # får förstärkning.
    n_total = len(train_dataset_instance)
    if n_total == 0:
        print("Hittade ingen data. Avslutar.")
        return
        
    indices = list(range(n_total))
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    
    # Blanda indexen (sätt ett seed för repeterbarhet)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    shuffled_indices = torch.randperm(n_total)
    
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    
    # Skapa "Subsets"
    # train_ds använder datasetet MED förstärkning
    train_ds = Subset(train_dataset_instance, train_indices)
    # val_ds använder datasetet UTAN förstärkning
    val_ds = Subset(val_dataset_instance, val_indices)
    # --- SLUT PÅ ÄNDRING ---

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, # Vi kan blanda här eftersom indexen redan är slumpade
        collate_fn=pad_collate_fn,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=pad_collate_fn,
        pin_memory=True,
        num_workers=0
    )

    # 3) Modell
    # Vi måste hämta label-mappningen från en av instanserna
    label_to_idx = train_dataset_instance.label_to_idx
    idx_to_label = train_dataset_instance.idx_to_label
    
    model = WordClassifier(n_mels=args.n_mels, n_classes=len(label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    print("\nDEBUG: Allt är laddat. PÅBÖRJAR TRÄNINGSLOOPEN NU...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mels": args.n_mels,
                    "label_to_idx": label_to_idx, # Använd sparad mappning
                    "idx_to_label": idx_to_label, # Använd sparad mappning
                },
                args.output,
            )
            print(f"  -> New best model saved to {args.output}")

    print("Training finished.")


if __name__ == "__main__":
    # Kör från projektroten:
    #   python -m src.training.train_word_classifier --data-root data/wav
    main()