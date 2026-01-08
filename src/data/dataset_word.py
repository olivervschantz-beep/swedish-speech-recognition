# src/data/dataset_word.py
import os
from typing import Callable, Optional, List, Tuple

import torch
import torchaudio # <--- NY IMPORT
import torchaudio.transforms as T # <--- NY IMPORT
from torch.utils.data import Dataset
import librosa
import random # <--- NY IMPORT


class WordDataset(Dataset):
    def __init__(
        self, 
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        data_dirs: List[str] = None,
        target_sample_rate: int = 16000,
        apply_augmentation: bool = False # <--- NY PARAMETER
    ):
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.data: List[Tuple[str, int]] = []
        
        # --- NY DEL: Augmentation ---
        self.apply_augmentation = apply_augmentation
        print(f"Dataset skapat med augmentation = {self.apply_augmentation}")
        # --- SLUT NY DEL ---
        
        
        # mappnamn -> klassindex
        self.label_to_idx = {"sju": 0, "korsord": 1, "kraftskiva": 2}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        # --- SKAPA TRANSFORMS EN GÅNG ---
        # Skapa en lista med alla "pitch shift"-maskiner vi vill ha
        self.pitch_shifters = [
            T.PitchShift(self.target_sample_rate, n_steps=-2),
            T.PitchShift(self.target_sample_rate, n_steps=-1),
            T.PitchShift(self.target_sample_rate, n_steps=1),
            T.PitchShift(self.target_sample_rate, n_steps=2),
        ]
        print(f"Skapat 4 pitch shift-objekt för augmentation")
        # --- SLUT PÅ NY KOD ---
        
        if data_dirs is None:
            data_dirs = [
                "data/wav/sju",
                "data/wav/korsord",
                "data/wav/kraftskiva",
            ]

        print("WordDataset letar i följande mappar:")
        for d in data_dirs:
            print("   -", os.path.abspath(d))
        
        valid_exts = {".wav"}

        for data_dir in data_dirs:
            label_name = os.path.basename(data_dir).lower()
            
            if label_name not in self.label_to_idx:
                print(f"Okänd label '{label_name}' i katalog {data_dir}")
                continue
                
            label = self.label_to_idx[label_name]
            
            if not os.path.exists(data_dir):
                print(f"Katalogen {data_dir} finns inte")
                continue

            file_list = os.listdir(data_dir)
            print(f"Hittade {len(file_list)} filer i {data_dir}")

            for filename in file_list:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in valid_exts:
                    continue
                    
                # Ignorera trasiga filer vi hittade tidigare
                if filename in ["AP_KO_3.wav", "AP_KO_4.wav", "AP_KO_5.wav"]:
                    continue

                file_path = os.path.join(data_dir, filename)
                self.data.append((file_path, label))
        
        print(f"Loaded {len(self.data)} audio files totalt")
        print(f"Class distribution (per label-index): {self.get_class_distribution()}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.data[idx]
        
        try:
        # Läs ljud med librosa (mer robust för konstiga filer)
            waveform_np, sample_rate = librosa.load(            
                file_path, 
                sr=self.target_sample_rate, 
                 mono=True
            )
             # Gör om från numpy-array -> torch-tensor [1, T]
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)

            # --- NY DEL: Augmentation ---
            # Tillämpa bara om flaggan är True
            if self.apply_augmentation:
            # 50% chans att ändra tonhöjd (med torch.rand)
            #    if torch.rand(1) > 0.5:
            #    # Välj en av de FÖRBEREDDA maskinerna (med torch.randint)
            #        idx = torch.randint(0, len(self.pitch_shifters), (1,))
            #        pitch_shifter = self.pitch_shifters[idx.item()]
            #        waveform = pitch_shifter(waveform)

            # 50% chans att lägga till brus (med torch.rand)
                if torch.rand(1) > 0.5:
                    # En liten mängd slumpmässigt brus (med torch.rand)
                    noise_level = (0.001 - 0.005) * torch.rand(1) + 0.005 
                    noise = torch.randn_like(waveform) * noise_level
                    waveform = waveform + noise
           
            # --- SLUT NY DEL ---

            if self.transform is not None:
                features = self.transform(waveform)   # [1, n_mels, time]
            else:
                features = waveform                   # [1, T]
            
            label_tensor = torch.tensor(label, dtype=torch.long)
            return features, label_tensor
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            if self.transform is not None:
                features = torch.zeros(1, 64, 100)
            else:
                features = torch.zeros(1, self.target_sample_rate)
                
            label_tensor = torch.tensor(label, dtype=torch.long)
            return features, label_tensor

    def get_class_distribution(self) -> dict:
        distribution = {label: 0 for label in self.label_to_idx.values()}
        for _, label in self.data:
            distribution[label] += 1
        return distribution

    def get_label_name(self, label_idx: int) -> str:
        return self.idx_to_label.get(label_idx, "UNKNOWN")