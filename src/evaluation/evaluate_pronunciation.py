# src/evaluation/evaluate_pronunciation.py

import argparse
import os
import torch
import torch.nn.functional as F  # <--- TILLAGD IMPORT
from torch.utils.data import DataLoader
import numpy as np
import argparse
# Importera era projekt-specifika moduler
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


# vvvvvv TILLAGD FUNKTION vvvvvv
def pad_collate_fn_pronunciation(batch):
    """
    Paddar spektrogram (features) till maxlängden i batchen.
    Denna batch innehåller inte labels.
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


def get_reconstruction_errors(
    model: PronunciationScorer,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> list[float]:
    """
    Kör all data i dataloadern genom modellen och returnera en
    lista med MSE-förlusten för varje enskild sample.
    """
    model.eval()
    all_errors = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            
            recon_features = model(features)

            # Padda/klipp recon_features för att matcha features exakta storlek
            input_size = features.shape[2:]
            recon_features = recon_features[:, :, :input_size[0], :input_size[1]]
            
            # Beräkna fel för varje sample i batchen
            for i in range(features.size(0)):
                # [B, C, n_mels, time] -> [C, n_mels, time]
                f_sample = features[i]
                r_sample = recon_features[i]
                
                loss = criterion(r_sample, f_sample).item()
                all_errors.append(loss)
                
    return all_errors


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pronunciation models (autoencoders) and find thresholds."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Mapp som innehåller de tränade modellerna (t.ex. pronunciation_scorer_sju.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/test", # <--- BÖR vara en separat test-mapp!
        help="Rotmapp som innehåller testdata (sju/, korsord/, kraftskiva/)",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Antal mels (måste matcha det som användes vid träning)"
    )
    parser.add_argument(
        "--n-std",
        type=float,
        default=3.0,
        help="Antal standardavvikelser över medelvärdet som tröskel."
    )

    args = parser.parse_args()
    device = get_device()
    print(f"Using device: {device}")
    
    words = ["sju", "korsord", "kraftskiva"]
    thresholds = {}
    all_error_stats = {} # För att spara medelvärde/std

    # 1. Förbered transform och förlustfunktion
    transform = make_log_mel_transform(
        sample_rate=16_000,
        n_mels=args.n_mels,
    )
    criterion = torch.nn.MSELoss(reduction='mean')

    print("="*50)
    print("Del 1: Beräkna tröskelvärden från korrekt testdata")
    print("="*50)

    # 2. Ladda varje modell och beräkna fel-distributionen för dess korrekta data
    for word in words:
        print(f"\n--- Utvärderar modell för: '{word}' ---")
        
        # Ladda modell
        model_path = os.path.join(args.model_dir, f"pronunciation_scorer_{word}.pt")
        if not os.path.exists(model_path):
            print(f"Modellfil hittades inte: {model_path}")
            continue
            
        checkpoint = torch.load(model_path, map_location=device)
        model = PronunciationScorer(n_mels=args.n_mels).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Ladda endast korrekt data för detta ord
        data_dir = os.path.join(args.data_root, word)
        if not os.path.exists(data_dir):
            print(f"Datamapp hittades inte: {data_dir}")
            continue

        dataset = PronunciationDataset(
            data_dir=data_dir,
            transform=transform,
            target_sample_rate=16_000
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            shuffle=False,
            collate_fn=pad_collate_fn_pronunciation  # <--- TILLAGD RAD
        )

        # Beräkna alla fel
        errors = get_reconstruction_errors(model, dataloader, criterion, device)
        
        if not errors:
            print(f"Hittade ingen data i {data_dir}")
            continue

        # Beräkna statistik
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        # Beräkna tröskel
        threshold = mean_error + (args.n_std * std_error)
        
        thresholds[word] = threshold
        all_error_stats[word] = {'mean': mean_error, 'std': std_error}

        print(f"Resultat för {len(errors)} '{word}' testfiler:")
        print(f"  Medelfel (MSE): {mean_error:.6f}")
        print(f"  Std-avvikelse:  {std_error:.6f}")
        print(f"  Max-fel:        {max_error:.6f}")
        print(f"  REKOMMENDERAD TRÖSKEL (Mean + {args.n_std}*Std): {threshold:.6f}")

    print("\n" + "="*50)
    print("Del 2: Korsvalidering (Testa fel ord mot modellerna)")
    print("="*50)
    print("Detta testar om 'fel' ord ger ett högre fel än tröskeln.")
    print("Värdena nedan är Medel-MSE. Högre är bättre.\n")

    # Skriv ut tabell-header
    header = f"{'Modell':<12} | " + " | ".join([f"Data: {w:<10}" for w in words])
    print(header)
    print("-" * len(header))

    # 3. Korsvalidera: kör "fel" data genom varje modell
    for model_word in words:
        row = f"{model_word:<12} | "
        
        # Ladda modell
        model_path = os.path.join(args.model_dir, f"pronunciation_scorer_{model_word}.pt")
        if not os.path.exists(model_path): continue
        
        checkpoint = torch.load(model_path, map_location=device)
        model = PronunciationScorer(n_mels=args.n_mels).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        for data_word in words:
            # Ladda data
            data_dir = os.path.join(args.data_root, data_word)
            if not os.path.exists(data_dir):
                row += f"{'N/A':<10} | "
                continue

            dataset = PronunciationDataset(
                data_dir=data_dir,
                transform=transform,
                target_sample_rate=16_000
            )
            dataloader = DataLoader(
                dataset, 
                batch_size=16, 
                shuffle=False,
                collate_fn=pad_collate_fn_pronunciation  # <--- TILLAGD RAD
            )
            
            errors = get_reconstruction_errors(model, dataloader, criterion, device)
            mean_error = np.mean(errors) if errors else 0.0
            
            # Formatera och lägg till i raden
            row += f"{mean_error:<10.6f} | "
        
        print(row)

    print("\n" + "="*50)
    print("SAMMANFATTNING: REKOMMENDERADE TRÖSKELVÄRDEN")
    print("="*50)
    print("Kopiera dessa värden till ert `predict.py` skript:")
    print(thresholds)


if __name__ == "__main__":
    # Kör från projektroten:
    #   python -m src.evaluation.evaluate_pronunciation --data-root data/test
    main()