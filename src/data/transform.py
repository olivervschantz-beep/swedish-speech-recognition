# src/data/transforms.py
from typing import Callable

import torch
import torchaudio


def make_log_mel_transform(
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Callable[[torch.Tensor], torch.Tensor]:
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
        center=True,
        pad_mode="reflect",
        norm=None,
    )

    def transform(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform_mono = waveform.mean(dim=0, keepdim=True)
        else:
            waveform_mono = waveform  # [1, T]

        mel_spec = mel_spec_transform(waveform_mono)  # [1, n_mels, time_frames]
        log_mel = torch.log(mel_spec + 1e-6)
        return log_mel
    
    return transform