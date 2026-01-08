# src/models/pronunciation_scorer.py

import torch
import torch.nn as nn

class PronunciationScorer(nn.Module):
    """
    En Convolutional Autoencoder (CAE) designad för att lära sig
    att rekonstruera log-mel-spektrogram.
    
    Modellen tränas *endast* på "korrekta" uttal. Vid inferens,
    om rekonstruktionsfelet (t.ex. MSE) mellan input och output
    är högt, klassas uttalet som "ej korrekt" (en anomali).
    """
    def __init__(self, n_mels: int = 64):
        """
        Initierar Encoder- och Decoder-delarna.
        
        Args:
            n_mels (int): Antalet mel-frekvensband i spektrogrammet (höjden).
        """
        super().__init__()
        
        # Vi antar att n_mels är t.ex. 64. Tidsdimensionen (bredden) kan variera.
        # Vi använder `AdaptiveAvgPool2d` för att hantera varierande längd
        # och få en fast storlek innan vi går in i de linjära lagren.

        # --- ENCODER ---
        # Input: (batch_size, 1, n_mels, n_frames)
        self.encoder = nn.Sequential(
            # Conv 1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # <--- ÄNDRAD
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 32, ...)
            
            # Conv 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # <--- ÄNDRAD
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (B, 64, ...)
            
            # Conv 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # <--- ÄNDRAD
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (B, 128, ...)
        )
        
        # --- DECODER ---
        # Input: (B, 64, n_mels/8, n_frames/8)
        self.decoder = nn.Sequential(
            # Transposed Conv 1
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # <--- ÄNDRAD
            nn.ReLU(True),
            
            # Transposed Conv 2
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), # <--- ÄNDRAD
            nn.ReLU(True),
            
            # Transposed Conv 3
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2) # <--- ÄNDRAD
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kör input-spektrogrammet (x) genom autoencodern.
        
        Args:
            x (torch.Tensor): Input-tensor (spektrogram). 
                              Förväntad form: (batch_size, 1, n_mels, n_frames)
                              
        Returns:
            torch.Tensor: Rekonstruerad tensor (x_recon).
                          Form: (batch_size, 1, n_mels, n_frames)
        """
        
        # Vi måste se till att inputen har en kanaldimension
        if x.dim() == 3:
            # Form (B, n_mels, n_frames) -> (B, 1, n_mels, n_frames)
            x = x.unsqueeze(1)
            
        z = self.encoder(x)
        x_recon = self.decoder(z)
        
        # Se till att output-storleken matchar input-storleken
        # (ConvTranspose2d kan ge +/- 1 pixel pga. avrundning)
        # Vi klipper/paddar 'x_recon' till samma storlek som 'x'
        
        # Ta hänsyn till eventuella storleksskillnader efter upsampling
        # (Detta är en robusthetsåtgärd)
        input_size = x.shape[2:] # (n_mels, n_frames)
        output_size = x_recon.shape[2:]
        
        pad_h = input_size[0] - output_size[0]
        pad_w = input_size[1] - output_size[1]
        
        # Padding (vänster, höger, topp, botten)
        # Vi lägger bara till padding på höger/botten
        padding = (0, max(0, pad_w), 0, max(0, pad_h))
        x_recon = nn.functional.pad(x_recon, padding)
        
        # Klippning
        x_recon = x_recon[:, :, :input_size[0], :input_size[1]]
        
        return x_recon