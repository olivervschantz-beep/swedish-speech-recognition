# src/data/dataset_pronunciation.py
import os
from typing import Callable, Optional, List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset
import librosa


class PronunciationDataset(Dataset):
    """
    Dataset för att ladda ljudfiler för en *enda* klass,
    används för att träna autoencodern (PronunciationScorer).
    
    Denna klass returnerar *endast* features, eftersom målet
    är att rekonstruera inputen (features -> model -> features_recon).
    """
    def __init__(
        self, 
        data_dir: str, # Tar bara EN mapp, t.ex. "data/wav/sju"
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_sample_rate: int = 16000,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.file_paths: List[str] = []
        
        valid_exts = {".wav"}

        if not os.path.exists(data_dir):
            print(f"Katalogen {data_dir} finns inte")
            return

        file_list = os.listdir(data_dir)
        print(f"PronunciationDataset letar i: {os.path.abspath(data_dir)}")
        print(f"Hittade {len(file_list)} filer.")

        for filename in file_list:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_exts:
                continue

            file_path = os.path.join(self.data_dir, filename)
            self.file_paths.append(file_path)
        
        print(f"Loaded {len(self.file_paths)} audio files totalt")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returnerar:
            features (torch.Tensor): De transformerade ljuddata (t.ex. log-mel).
        """
        file_path = self.file_paths[idx]
        
        

            # 2. VI HOPPAR ÖVER OMSAMPLING. Vi litar på att filerna är 16k.
            # Att anropa T.Resample här inne orsakar frysning-buggen.
        try:
            # Läs ljud med librosa (mer robust för konstiga filer)
            waveform_np, sample_rate = librosa.load(
                file_path, 
                sr=self.target_sample_rate, 
                mono=True
            )
            # Gör om från numpy-array -> torch-tensor [1, T]
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)    

            # Transform → features (log-mel)
            if self.transform is not None:
                features = self.transform(waveform)   # [1, n_mels, time]
            else:
                features = waveform                   # [1, T]
            
            # Ta bort kanal-dimensionen om den finns, 
            # modellen och transformen hanterar detta.
            # features: [1, n_mels, time] -> [n_mels, time]
            features = features.squeeze(0) 
            
            return features
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            if self.transform is not None:
                # Returnera en tom tensor med rätt antal mels
                # (Antar 64 mels från train_word_classifier)
                return torch.zeros(64, 100) 
            else:
                return torch.zeros(self.target_sample_rate)
            