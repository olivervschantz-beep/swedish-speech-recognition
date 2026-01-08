# src/training/train_pronunciation.py

import argparse
import os

import torch
import torch.nn.functional as F  # <--- TILLAGD IMPORT
from torch.utils.data import DataLoader, random_split

from src.data.dataset_pronunciation import PronunciationDataset 
from src.data.transform import make_log_mel_transform
from src.models.pronunciation_scorer import PronunciationScorer


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
    total_samples = 0

    for features in dataloader:
        features = features.to(device)

        optimizer.zero_grad()
        
        recon_features = model(features)
        
        # Padda/klipp recon_features för att matcha features exakta storlek
        # (Kan behövas pga upsampling-artefakter)
        input_size = features.shape[2:]
        recon_features = recon_features[:, :, :input_size[0], :input_size[1]]

        loss = criterion(recon_features, features)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    avg_loss = total_loss / total_samples
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)

            recon_features = model(features)

            # Padda/klipp recon_features för att matcha features exakta storlek
            input_size = features.shape[2:]
            recon_features = recon_features[:, :, :input_size[0], :input_size[1]]
            
            loss = criterion(recon_features, features)

            batch_size = features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss


# vvvvvv TILLAGD FUNKTION vvvvvv
def pad_collate_fn_pronunciation(batch):
    """
    Paddar spektrogram (features) till maxlängden i batchen.
    Denna batch innehåller *inte* labels.
    """
    # batch är en lista av feature-tensorer med form [n_mels, time]
    features = batch 

    # Hitta maxlängden i tidsdimensionen (dim 1)
    max_len = max([f.shape[1] for f in features])

    padded_features = []
    for f in features:
        # Padda på höger sida (sista dimensionen)
        pad_width = max_len - f.shape[1]
        # (pad_vänster, pad_höger)
        padded_f = F.pad(f, (0, pad_width), "constant", 0)
        padded_features.append(padded_f)

    # Stacka de nu lika stora tensorerna
    # Resultat-form: [batch_size, n_mels, max_len]
    features_batch = torch.stack(padded_features)
    
    # Modellen förväntar sig [B, C=1, n_mels, time]
    features_batch = features_batch.unsqueeze(1)

    return features_batch
# ^^^^^^ SLUT PÅ TILLAGD FUNKTION ^^^^^^


def main():
    parser = argparse.ArgumentParser(
        description="Train Pronunciation Scorer (Autoencoder) for one word."
    )
    
    parser.add_argument(
        "--word",
        type=str,
        required=True,
        choices=["sju", "korsord", "kraftskiva"],
        help="Vilket ord som ska tränas."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/wav",
        help="Rotmapp som innehåller sju/, korsord/, kraftskiva/ (default: data/wav)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Mapp där den tränade modellen ska sparas."
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    
    word_name = args.word.lower()
    data_dir = os.path.join(args.data_root, word_name)
    output_path = os.path.join(args.output_dir, f"pronunciation_scorer_{word_name}.pt")
    
    print(f"Tränar modell för ordet: '{word_name}'")
    print(f"Datarot: {data_dir}")
    print(f"Modell sparas till: {output_path}")

    # 1) Transform (features)
    transform = make_log_mel_transform(
        sample_rate=16_000,
        n_mels=args.n_mels,
    )

    # 2) Dataset
    full_dataset = PronunciationDataset(
        data_dir=data_dir,
        transform=transform,
        target_sample_rate=16_000,
    )
    
    if len(full_dataset) == 0:
        print(f"Hittade inga filer i {data_dir}. Avslutar.")
        return

    n_total = len(full_dataset)
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=pad_collate_fn_pronunciation,  # <--- TILLAGD RAD
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=pad_collate_fn_pronunciation,  # <--- TILLAGD RAD
        num_workers=0,
        pin_memory=True
    )

    # 3) Modell
    model = PronunciationScorer(n_mels=args.n_mels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    criterion = torch.nn.MSELoss() 

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "n_mels": args.n_mels,
                },
                output_path,
            )
            print(f"  -> New best model saved to {output_path} (loss: {val_loss:.6f})")

    print("Training finished.")


if __name__ == "__main__":
    # Kör från projektroten, t.ex.:
    #   python -m src.training.train_pronunciation --word sju
    main()
    