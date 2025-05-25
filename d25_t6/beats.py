import torch
import torch.nn as nn
import torchaudio
import os

class BEATsWrapper(nn.Module):
    def __init__(self, checkpoint_path, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Load BEATs model definition from torch.hub
        self.model = torch.hub.load(
            'microsoft/BEATs',
            'BEATs',
            source='github',
            trust_repo=True
        )
        # Load fine-tuned checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(self.device)
        self.target_sr = 16000

    def forward(self, x):
        # x: (batch, time), float32, [-1, 1], any sample rate
        # If input is not 16kHz, resample
        if hasattr(x, 'sr') and x.sr != self.target_sr:
            x = torchaudio.functional.resample(x, x.sr, self.target_sr)
        elif x.shape[-1] != self.target_sr and x.shape[-1] % self.target_sr != 0:
            # Assume input is 32kHz, resample to 16kHz
            x = torchaudio.functional.resample(x, 32000, self.target_sr)
        x = x.to(self.device)
        with torch.no_grad():
            # BEATs expects (batch, time)
            features = self.model.extract_features(x)[0]  # (batch, frames, 768)
            pooled = features.mean(dim=1)  # (batch, 768)
        return pooled 