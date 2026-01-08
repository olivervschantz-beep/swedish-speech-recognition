import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# ---- KORREKTA IMPORTER ----
from ..models.pronunciation_scorer import PronunciationScorer
from ..data.transform import make_log_mel_transform
# ---------------------------

# Se till att matplotlib fungerar på Mac
import matplotlib
matplotlib.use('TkAgg') 

def plot_reconstruction(model, transform_fn, file_path, device, title=""):
    """
    Laddar en ljudfil, kör den genom autoencodern och plottar
    Original vs. Rekonstruktion vs. Felkarta.
    
    'transform_fn' är en FUNKTION som körs på CPU.
    """
    
    # 1. Ladda ljudfil (på CPU)
    target_sample_rate = 16000
    waveform_np, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    # Skapa CPU-tensor
    waveform_t_cpu = torch.from_numpy(waveform_np).float().unsqueeze(0) 

    # 2. Skapa spektrogram (Input) - på CPU
    # Detta matchar träningskoden, där transformen sker FÖRE .to(device)
    spec_cpu = transform_fn(waveform_t_cpu) # [1, n_mels, time]

    # 3. Flytta till device och förbered för modell
    spec = spec_cpu.to(device) # <-- Flytta till device HÄR
    spec_batch = spec.unsqueeze(1) # [1, 1, n_mels, time]

    # 4. Kör genom modellen (Output)
    model.eval()
    with torch.no_grad():
        recon_spec_batch = model(spec_batch) # [1, 1, n_mels, time]

    # 5. Beräkna fel
    # Flytta tillbaka till CPU för numpy/plotting
    spec_np = spec.squeeze().cpu().numpy()
    recon_np = recon_spec_batch.squeeze().cpu().numpy()
    
    mse = np.mean((spec_np - recon_np)**2)
    error_map_np = np.abs(spec_np - recon_np)
    
    print(f"Fil: {os.path.basename(file_path)}, MSE Loss: {mse:.4f}")

    # 6. Plotta allt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"{title}\n{os.path.basename(file_path)} (MSE: {mse:.4f})", fontsize=16)

    im1 = ax1.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title("Original Spectrogram")
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(recon_np, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title("Reconstructed by Model")
    fig.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(error_map_np, aspect='auto', origin='lower', cmap='hot')
    ax3.set_title("Error Map (Difference)")
    fig.colorbar(im3, ax=ax3)

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Visualisera en Autoencoders rekonstruktion.")
    parser.add_argument("--word", type=str, required=True, help="Ordet som ska testas (t.ex. 'sju')")
    parser.add_argument("--good-file", type=str, required=True, help="Sökväg till en 'bra' ljudfil.")
    parser.add_argument("--bad-file", type=str, required=True, help="Sökväg till en 'dålig' ljudfil.")
    
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Ladda modellen
    model_path = f"models/pronunciation_scorer_{args.word}.pt"
    if not os.path.exists(model_path):
        print(f"Hittade inte modellen: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=device)
    n_mels = checkpoint.get("n_mels", 64) # Använd 64 som fallback
        
    model = PronunciationScorer(n_mels=n_mels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Modell laddad från {model_path} (n_mels={n_mels})")

    # 2. Skapa transformen
    sample_rate = 16000
    
    # ---- KORREKT TRANSFORM ----
    # Vi skapar funktionen, men anropar INTE .to(device) på den.
    transform_fn = make_log_mel_transform(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    print(f"Ljud-transform-funktion skapad (sample_rate={sample_rate}, n_mels={n_mels})")

    # 3. Plotta för den bra filen
    # Vi skickar 'transform_fn' till plot-funktionen
    plot_reconstruction(model, transform_fn, args.good_file, device, title="Test av KORREKT uttal")
    
    # 4. Plotta för den dåliga filen
    plot_reconstruction(model, transform_fn, args.bad_file, device, title="Test av FELAKTIGT uttal")

    # Visa båda plot-fönstren
    plt.show()

if __name__ == "__main__":
    main()


# För att köra: 
# python3 -m src.evaluation.visualize_autoencoder \
#    --word sju \
#    --good-file data/test/sju/sju_bra_1.wav \
#    --bad-file data/test/sju/sju_dålig_1.wav