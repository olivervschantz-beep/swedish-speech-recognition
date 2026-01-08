# inference/predict.py

import torchaudio
import argparse
import os
import torch
import librosa
import numpy as np

# Importera era projekt-specifika moduler
from models.word_classifier import WordClassifier
from models.pronunciation_scorer import PronunciationScorer
from data.transform import make_log_mel_transform

# ###################################################################
THRESHOLDS = {
    'sju': 8.5, #Bör uppdateras!
    'korsord': 8.5, #Bör uppdateras!
    'kraftskiva': 8.5 #Bör uppdateras!
}
# ###################################################################


def get_device() -> torch.device:
    """Välj GPU om möjligt, annars CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Mac med Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_all_models(model_dir: str, n_mels: int, device: torch.device):
    """
    Laddar in ordklassificeraren och alla tre autoencoder-modeller.
    """
    models = {}
    
    # 1. Ladda Word Classifier
    print("Laddar word_classifier.pt...")
    wc_path = os.path.join(model_dir, "word_classifier.pt")
    if not os.path.exists(wc_path):
        raise FileNotFoundError(f"Kunde inte hitta {wc_path}")
        
    checkpoint = torch.load(wc_path, map_location=device)
    n_classes = len(checkpoint['idx_to_label'])
    
    word_model = WordClassifier(n_mels=n_mels, n_classes=n_classes).to(device)
    word_model.load_state_dict(checkpoint['model_state'])
    word_model.eval()
    
    models['word_classifier'] = word_model
    models['idx_to_label'] = {int(k): v for k, v in checkpoint['idx_to_label'].items()}
    
    # 2. Ladda Autoencoders
    ae_models = {}
    for word in ["sju", "korsord", "kraftskiva"]:
        print(f"Laddar pronunciation_scorer_{word}.pt...")
        ae_path = os.path.join(model_dir, f"pronunciation_scorer_{word}.pt")
        if not os.path.exists(ae_path):
            print(f"Varning: Kunde inte hitta {ae_path}")
            continue
        
        ae_checkpoint = torch.load(ae_path, map_location=device)
        ae_model = PronunciationScorer(n_mels=n_mels).to(device)
        ae_model.load_state_dict(ae_checkpoint['model_state'])
        ae_model.eval()
        ae_models[word] = ae_model
        
    models['autoencoders'] = ae_models
    return models

def load_audio_and_transform(
    file_path: str,
    transform: callable,
    sample_rate: int,
    device: torch.device
) -> torch.Tensor:
    """Läs en ljudfil, transformera och förbered för modellen."""
    
    # Läs ljud med librosa
    waveform_np, sr = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True,
    )
    
    # Gör om till torch [1, T]
    waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)

    # Transform -> features (log-mel) [1, n_mels, time]
    features = transform(waveform)
    
    # Skicka till device och lägg till batch-dimension [B=1, C=1, n_mels, time]
    features = features.to(device)
    if features.dim() == 3: # Saknar kanal-dimension
        features = features.unsqueeze(1)
        
    return features

def run_prediction(
    file_path: str,
    model_dir: str = "models",
    n_mels: int = 64,
    sample_rate: int = 16000,
) -> dict:
    """
    Kör samma logik som main(), men returnerar ett resultat-objekt istället
    för att bara skriva ut text.

    Returnerar en dict:
      {
        'file': ...,
        'predicted_word': ...,
        'pronunciation_label': 'Correct' / 'Incorrect' / 'N/A',
        'error': float,
        'threshold': float | None,
        'is_correct': bool | None,
      }
    """
    device = get_device()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Ladda alla modeller
    all_models = load_all_models(model_dir, n_mels, device)
    word_model = all_models["word_classifier"]
    idx_to_label = all_models["idx_to_label"]
    autoencoder_models = all_models["autoencoders"]

    # 2. Ladda + transformera ljud
    transform = make_log_mel_transform(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )
    features = load_audio_and_transform(file_path, transform, sample_rate, device)

    criterion = torch.nn.MSELoss()

    # 3. Ordklassificering
    with torch.no_grad():
        logits = word_model(features)
        pred_idx = logits.argmax(dim=1).item()
        predicted_word = idx_to_label[pred_idx]

    # 4. Uttalsbedömning
    pronunciation_label = "N/A"
    error = 0.0
    threshold_to_use = None
    is_correct = None

    if predicted_word in autoencoder_models:
        model_to_use = autoencoder_models[predicted_word]
        threshold_to_use = THRESHOLDS[predicted_word]

        with torch.no_grad():
            recon_features = model_to_use(features)
            # Säkerställ att shapes matchar
            recon_features = recon_features[:, :, :features.size(2), :features.size(3)]
            error = criterion(recon_features, features).item()

        if error <= threshold_to_use:
            pronunciation_label = "Korrekt"
            is_correct = True
        else:
            pronunciation_label = "Ej Korrekt (anomali)"
            is_correct = False
    else:
        print(f"Varning: Ingen uttalsscorer-modell hittades för '{predicted_word}'.")

    return {
        "file": file_path,
        "predicted_word": predicted_word,
        "pronunciation_label": pronunciation_label,
        "error": float(error),
        "threshold": float(threshold_to_use) if threshold_to_use is not None else None,
        "is_correct": is_correct,
    }
def main():
    parser = argparse.ArgumentParser(
        description="Gör en fullständig prediktion (ord + uttal) på en .wav-fil."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Sökväg till den .wav-fil som ska testas."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Mapp som innehåller alla tränade modeller."
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=64,
        help="Antal mels (måste matcha det som användes vid träning)."
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate (måste matcha det som användes vid träning)."
    )

    args = parser.parse_args()
    device = get_device()
    
    if not os.path.exists(args.file):
        print(f"Fil hittades inte: {args.file}")
        return

    # 1. Ladda alla modeller i minnet
    try:
        all_models = load_all_models(args.model_dir, args.n_mels, device)
        word_model = all_models['word_classifier']
        idx_to_label = all_models['idx_to_label']
        autoencoder_models = all_models['autoencoders']
    except FileNotFoundError as e:
        print(e)
        print("Avslutar. Se till att alla modeller finns i", args.model_dir)
        return

    # 2. Ladda och transformera ljudfilen
    transform = make_log_mel_transform(
        sample_rate=args.sr,
        n_mels=args.n_mels,
    )
    features = load_audio_and_transform(args.file, transform, args.sr, device)
    
    # Förlustfunktion för att mäta fel
    criterion = torch.nn.MSELoss()

    # 3. Steg 1: Prediktera vilket ord det är
    with torch.no_grad():
        logits = word_model(features)
        pred_idx = logits.argmax(dim=1).item()
        predicted_word = idx_to_label[pred_idx]

    # 4. Steg 2: Bedöm uttalet med rätt autoencoder
    pronunciation_score = "N/A"
    error = 0.0
    threshold = 0.0
    
    if predicted_word in autoencoder_models:
        model_to_use = autoencoder_models[predicted_word]
        threshold_to_use = THRESHOLDS[predicted_word]
        
        with torch.no_grad():
            recon_features = model_to_use(features)
            
            # Se till att storlekarna matchar exakt för loss-beräkning
            # (Modellen kan ha klippt/paddat output)
            recon_features = recon_features[:, :, :features.size(2), :features.size(3)]
            
            error = criterion(recon_features, features).item()
        
        if error <= threshold_to_use:
            pronunciation_score = "Correkt"
        else:
            pronunciation_score = "Incorrect"
            
    else:
        print(f"Warning: No pronounciation scorer model found'{predicted_word}'.")


    # 5. Skriv ut det slutgiltiga resultatet
    print("\n" + "="*30)
    print("      PREDIKTIONSRESULTAT      ")
    print("="*30)
    print(f"Fil: {args.file}\n")
    print(f"  -> Predikterat ord: {predicted_word}")
    print(f"  -> Uttalsbedömning:  {pronunciation_score}")
    print(f"     (Fel: {error:.6f}, Tröskel: {threshold_to_use:.6f})")


if __name__ == "__main__":
    # Kör från projektroten, t.ex.:
    #   python -m inference.predict --file data/test/sju/min_nya_fil.wav
    #   python -m inference.predict --file data/test/korsord/ett_annat_test.wav
    
    main()