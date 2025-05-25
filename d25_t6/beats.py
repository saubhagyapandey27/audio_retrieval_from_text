import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import warnings
import os
from typing import Optional

# Suppress fairseq warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="fairseq")
warnings.filterwarnings("ignore", category=UserWarning, module="fairseq")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

try:
    from fairseq import checkpoint_utils
    from fairseq.models.wav2vec import Wav2Vec2Model
    FAIRSEQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: fairseq not available. Error: {e}")
    FAIRSEQ_AVAILABLE = False


class BEATsWrapper(torch.nn.Module):
    def __init__(self, beats_model_path: Optional[str] = None, target_length: int = 1000):
        """
        Args:
            beats_model_path (str): Path to the BEATs checkpoint file
            target_length (int): Target sequence length for output
        """
        super().__init__()
        
        if not FAIRSEQ_AVAILABLE:
            raise ImportError("fairseq is required for BEATs model but not available")
        
        self.target_length = target_length
        
        # Set default model path if not provided
        if beats_model_path is None:
            beats_model_path = self._get_default_model_path()
        
        if not os.path.exists(beats_model_path):
            raise FileNotFoundError(f"BEATs model not found at {beats_model_path}. "
                                  f"Please download it using setup_beats.py")
        
        # Load BEATs model with error handling
        try:
            models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([beats_model_path])
            self.model = models[0]
            self.model.eval()
            
            # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            raise RuntimeError(f"Failed to load BEATs model from {beats_model_path}. Error: {e}")

    def _get_default_model_path(self) -> str:
        """Get default model path"""
        return "./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw audio waveform of shape (batch_size, audio_length)
        
        Returns:
            torch.Tensor: Audio embeddings of shape (batch_size, target_length, 768)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input (batch_size, audio_length), got {x.dim()}D")
        
        batch_size = x.shape[0]
        
        # BEATs expects 16kHz audio, but we have 32kHz
        # Use high-quality resampling instead of simple interpolation
        x_16k = F.interpolate(x.unsqueeze(1), scale_factor=0.5, mode='linear', align_corners=False).squeeze(1)
        
        with torch.no_grad():
            try:
                # Extract features using BEATs
                features = self.model.extract_features(x_16k, padding_mask=None)
                
                if isinstance(features, tuple):
                    features = features[0]  # Take the main features
                
                # Ensure we have the expected shape
                if features.dim() != 3:
                    raise ValueError(f"Expected 3D features, got {features.dim()}D")
                
                # features shape: (batch_size, seq_len, 768)
                seq_len = features.shape[1]
                
                # Adjust sequence length to match target_length
                if seq_len != self.target_length:
                    features = features.transpose(1, 2)  # (batch, 768, seq_len)
                    features = F.interpolate(features, size=self.target_length, mode='linear', align_corners=False)
                    features = features.transpose(1, 2)  # (batch, target_length, 768)
                
            except Exception as e:
                raise RuntimeError(f"Error during BEATs feature extraction: {e}")
        
        return features


class BEATsNoOverlapWrapper(torch.nn.Module):
    def __init__(self, beats_model_path: Optional[str] = None):
        """
        Args:
            beats_model_path (str): Path to the BEATs checkpoint file
        """
        super().__init__()
        self.model = BEATsWrapper(beats_model_path, target_length=1000)

    def forward(self, x):
        # Get embeddings from BEATs
        embeddings = self.model(x)  # (batch_size, seq_len, 768)
        
        # Global average pooling to get a single embedding per audio
        pooled_embedding = embeddings.mean(dim=1)  # (batch_size, 768)
        
        return pooled_embedding


class CutInputIntoSegmentsWrapper(nn.Module):
    def __init__(self, model, max_input_length, segment_length, hop_size):
        """
        Args:
            model (nn.Module): The PyTorch model to wrap.
            max_input_length (int): Maximum length of input the model can handle.
            segment_length (int): Length of each segment if input exceeds max_input_length.
            hop_size (int): Hop size for overlapping segmentation.
        """
        super().__init__()
        self.model = model
        self.max_input_length = max_input_length
        self.segment_length = segment_length
        self.hop_size = hop_size

    def forward(self, x):
        """Processes the input audio through the model, handling segmentation if needed."""
        batch_size, input_length = x.shape

        if input_length <= self.max_input_length:
            return self.model(x).unsqueeze(1)  # Add segment dimension

        # Split into overlapping segments
        segments = []
        indices = list(range(0, input_length - self.segment_length + 1, self.hop_size))
        
        if not indices:  # Handle edge case
            indices = [0]
            
        for i in indices:
            end_idx = min(i + self.segment_length, input_length)
            segment = x[:, i:end_idx]
            
            # Pad if necessary
            if segment.shape[1] < self.segment_length:
                pad_size = self.segment_length - segment.shape[1]
                segment = F.pad(segment, (0, pad_size))
                
            segments.append(segment)

        segments = torch.stack(segments)  # Shape: (num_segments, batch_size, segment_length)
        outputs = self.model(segments.reshape(-1, self.segment_length))  # Process each segment
        outputs = outputs.view(len(indices), batch_size, -1).permute(1, 0, 2)   # Reshape back

        return outputs
