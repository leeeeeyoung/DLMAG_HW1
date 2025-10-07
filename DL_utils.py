"""
Shared utilities for DL artist classification training and inference pipeline.
Contains common classes, functions, and configurations used by both training and inference scripts.
"""

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
import multiprocessing
import json

warnings.filterwarnings("ignore")

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import top_k_accuracy_score

# ------------------ CPU / Thread tuning (set before heavy imports for best effect) ------------------
_cpu_count = multiprocessing.cpu_count()
_workers = max(1, _cpu_count // 2)  # Use half of CPU cores

# Set BLAS / OMP / MKL / OPENBLAS thread count (better set before numpy / torch imports)
os.environ.setdefault("OMP_NUM_THREADS", str(_workers))
os.environ.setdefault("MKL_NUM_THREADS", str(_workers))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_workers))
os.environ.setdefault("AUDIOREAD_BACKEND", "ffmpeg")  # Keep original ffmpeg setting

# after importing torch, set PyTorch thread config
torch.set_num_threads(_workers)
torch.set_num_interop_threads(1)

# ----------------------------- Config --------------------------------

class Config:
    """Configuration settings for the artist classification system."""
    SR: int = 16000  # Sample rate
    N_MELS: int = 320  # Number of mel bins
    N_FFT: int = 1024  # FFT window size
    HOP: int = 160  # Hop length
    N_MFCC: int = 40  # Number of MFCC coefficients
    SEG_SEC: float = 30.0  # Audio segment duration in seconds
    TRAIN_SEG_PER_TRACK: int = 5  # Number of segments per track during training
    EVAL_SEGMENTS: int = 20  # Number of segments for evaluation
    BATCH: int = 16  # Batch size
    EPOCHS: int = 100  # Number of training epochs
    LR: float = 1e-4  # Learning rate
    SEED: int = 42  # Random seed

    # Paths
    DATA_ROOT = Path("./artist20")
    TRAIN_JSON = DATA_ROOT / "train.json"
    VAL_JSON = DATA_ROOT / "val.json"
    TEST_JSON = DATA_ROOT / "test.json"
    TEST_DIR = DATA_ROOT / "test"
    TEST_OUT = "test_pred_dl.json"
    BEST_MODEL = "checkpoint_dl.pth"
    LOG_DIR = Path("./logs")

cfg = Config()

# device & seeds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
random.seed(cfg.SEED)

# fixed mel-frame length
MAX_FRAMES = int(cfg.SR * cfg.SEG_SEC / cfg.HOP)

# -------------------------- Audio Utilities --------------------------

class AudioUtils:
    """Utility class for audio processing operations."""
    
    @staticmethod
    def ffmpeg_available() -> bool:
        """Check if ffmpeg and ffprobe are available."""
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    @staticmethod
    def ffmpeg_read(path: str, sr: int = cfg.SR, offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
        """Read audio file using ffmpeg with optional offset and duration."""
        if not AudioUtils.ffmpeg_available():
            raise RuntimeError("ffmpeg/ffprobe required. Install ffmpeg on your system.")
        
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_name = tmp.name
        tmp.close()
        
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if offset and offset > 0:
            cmd += ["-ss", f"{float(offset):.6f}"]
        cmd += ["-i", str(path)]
        if duration and duration > 0:
            cmd += ["-t", f"{float(duration):.6f}"]
        cmd += ["-ac", "1", "-ar", str(sr), tmp_name, "-y"]
        
        subprocess.run(cmd, check=True)
        ret, _ = librosa.load(tmp_name, sr=sr, mono=True)
        
        try:
            os.remove(tmp_name)
        except Exception:
            pass
        
        return ret

    @staticmethod
    def get_duration(path: str) -> float:
        """Get audio file duration using ffprobe, fallback to loading entire file."""
        if AudioUtils.ffmpeg_available():
            try:
                output = subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", path],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                val = float(output)
                if val > 0:
                    return val
            except Exception:
                pass
        
        # fallback: load whole file
        ret = AudioUtils.ffmpeg_read(path, sr=cfg.SR)
        return float(len(ret)) / float(cfg.SR)

    @staticmethod
    def waveform_to_logmel(y: np.ndarray, sr: int = cfg.SR) -> np.ndarray:
        """Convert waveform to log mel-spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=cfg.N_MELS, n_fft=cfg.N_FFT, 
            hop_length=cfg.HOP, power=2.0
        )
        logmel = librosa.power_to_db(mel_spec + 1e-10).astype(np.float32)
        # normalize
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
        return logmel

    @staticmethod
    def extract_comprehensive_features(y: np.ndarray, sr: int = cfg.SR) -> np.ndarray:
        """Extract comprehensive audio features including log mel, MFCC, spectral contrast, and RMS energy."""
        # 1. Log mel-spectrogram (original feature)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=cfg.N_MELS, n_fft=cfg.N_FFT, 
            hop_length=cfg.HOP, power=2.0
        )
        logmel = librosa.power_to_db(mel_spec + 1e-10).astype(np.float32)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
        
        # 2. MFCC features (mean and std statistics)
        mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=cfg.N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Compute mean and std for MFCC features
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        mfcc_delta_mean = mfcc_delta.mean(axis=1)
        mfcc_delta_std = mfcc_delta.std(axis=1)
        mfcc_delta2_mean = mfcc_delta2.mean(axis=1)
        mfcc_delta2_std = mfcc_delta2.std(axis=1)
        
        # 3. Spectral contrast features (mean and std statistics)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP
        )
        spectral_contrast_mean = spectral_contrast.mean(axis=1)
        spectral_contrast_std = spectral_contrast.std(axis=1)
        
        # 4. RMS energy features (mean and std statistics)
        rms_energy = librosa.feature.rms(
            y=y, frame_length=cfg.N_FFT, hop_length=cfg.HOP
        )[0]
        rms_energy_mean = np.array([rms_energy.mean()])
        rms_energy_std = np.array([rms_energy.std()])
        
        # Get target dimensions from logmel
        target_frames = logmel.shape[1]
        target_mels = logmel.shape[0]
        
        # Create feature channels with consistent dimensions
        # All features will be resized to match logmel dimensions (target_mels x target_frames)
        
        # Resize MFCC features to match logmel dimensions
        mfcc_mean_resized = np.tile(mfcc_mean.reshape(-1, 1), (1, target_frames))
        mfcc_std_resized = np.tile(mfcc_std.reshape(-1, 1), (1, target_frames))
        mfcc_delta_mean_resized = np.tile(mfcc_delta_mean.reshape(-1, 1), (1, target_frames))
        mfcc_delta_std_resized = np.tile(mfcc_delta_std.reshape(-1, 1), (1, target_frames))
        mfcc_delta2_mean_resized = np.tile(mfcc_delta2_mean.reshape(-1, 1), (1, target_frames))
        mfcc_delta2_std_resized = np.tile(mfcc_delta2_std.reshape(-1, 1), (1, target_frames))
        
        # Resize spectral contrast features
        spectral_contrast_mean_resized = np.tile(spectral_contrast_mean.reshape(-1, 1), (1, target_frames))
        spectral_contrast_std_resized = np.tile(spectral_contrast_std.reshape(-1, 1), (1, target_frames))
        
        # Resize RMS energy features
        rms_energy_mean_resized = np.tile(rms_energy_mean.reshape(-1, 1), (1, target_frames))
        rms_energy_std_resized = np.tile(rms_energy_std.reshape(-1, 1), (1, target_frames))
        
        # Pad or truncate all features to match logmel frequency dimension
        def resize_to_target_mels(feature_array, target_mels):
            """Resize feature array to target mel dimension."""
            current_mels = feature_array.shape[0]
            if current_mels == target_mels:
                return feature_array
            elif current_mels < target_mels:
                # Pad with zeros
                pad_size = target_mels - current_mels
                return np.pad(feature_array, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            else:
                # Truncate
                return feature_array[:target_mels, :]
        
        # Resize all features to match logmel frequency dimension
        mfcc_mean_resized = resize_to_target_mels(mfcc_mean_resized, target_mels)
        mfcc_std_resized = resize_to_target_mels(mfcc_std_resized, target_mels)
        mfcc_delta_mean_resized = resize_to_target_mels(mfcc_delta_mean_resized, target_mels)
        mfcc_delta_std_resized = resize_to_target_mels(mfcc_delta_std_resized, target_mels)
        mfcc_delta2_mean_resized = resize_to_target_mels(mfcc_delta2_mean_resized, target_mels)
        mfcc_delta2_std_resized = resize_to_target_mels(mfcc_delta2_std_resized, target_mels)
        spectral_contrast_mean_resized = resize_to_target_mels(spectral_contrast_mean_resized, target_mels)
        spectral_contrast_std_resized = resize_to_target_mels(spectral_contrast_std_resized, target_mels)
        rms_energy_mean_resized = resize_to_target_mels(rms_energy_mean_resized, target_mels)
        rms_energy_std_resized = resize_to_target_mels(rms_energy_std_resized, target_mels)
        
        # Stack all features as channels
        comprehensive_features = np.stack([
            logmel,  # Channel 0: log mel-spectrogram
            mfcc_mean_resized,  # Channel 1: MFCC means
            mfcc_std_resized,  # Channel 2: MFCC stds
            mfcc_delta_mean_resized,  # Channel 3: MFCC-delta means
            mfcc_delta_std_resized,  # Channel 4: MFCC-delta stds
            mfcc_delta2_mean_resized,  # Channel 5: MFCC-delta2 means
            mfcc_delta2_std_resized,  # Channel 6: MFCC-delta2 stds
            spectral_contrast_mean_resized,  # Channel 7: Spectral contrast means
            spectral_contrast_std_resized,  # Channel 8: Spectral contrast stds
            rms_energy_mean_resized,  # Channel 9: RMS energy mean
            rms_energy_std_resized,  # Channel 10: RMS energy std
        ], axis=0)
        
        return comprehensive_features.astype(np.float32)

    @staticmethod
    def fix_length(mel: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
        """Fix mel-spectrogram length by padding or truncating."""
        if mel.shape[1] > max_len:
            return mel[:, :max_len]
        if mel.shape[1] < max_len:
            pad = max_len - mel.shape[1]
            return np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        return mel

    @staticmethod
    def fix_length_multi_channel(features: np.ndarray, max_len: int = MAX_FRAMES) -> np.ndarray:
        """Fix multi-channel feature length by padding or truncating."""
        if features.shape[2] > max_len:
            return features[:, :, :max_len]
        if features.shape[2] < max_len:
            pad = max_len - features.shape[2]
            return np.pad(features, ((0, 0), (0, 0), (0, pad)), mode="constant")
        return features

# --------------------------- Path helpers ---------------------------

def load_paths_from_json(json_path: Path) -> List[str]:
    """Load absolute file paths from JSON file."""
    base = json_path.parent
    with open(json_path, "r", encoding="utf-8") as fh:
        rel_paths = json.load(fh)
    
    out = []
    for p in rel_paths:
        norm_path = os.path.normpath(p)
        if norm_path.startswith("." + os.sep):
            norm_path = norm_path[2:]
        abs_path = norm_path if os.path.isabs(norm_path) else os.path.normpath(os.path.join(base, norm_path))
        out.append(abs_path)
    return out

def label_from_filepath(file_path: str) -> str:
    """Extract label (artist) from file path."""
    parts = os.path.normpath(file_path).split(os.sep)
    if "train_val" in parts:
        i = parts.index("train_val")
        if i + 1 < len(parts):
            return parts[i + 1]
    # fallback to parent folder name
    return parts[-2]

# ---------------------------- Datasets ------------------------------

class AudioDataset(Dataset):
    """Dataset class for audio data."""
    
    def __init__(self, file_list: List[str], class_list: List[str], mode: str = "train"):
        assert mode in ("train", "val"), "Mode must be 'train' or 'val'"
        self.mode = mode
        self.classes = class_list
        self.files = [f for f in file_list if os.path.isfile(f) and label_from_filepath(f) in self.classes]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        fp = self.files[idx]
        label = label_from_filepath(fp)
        label_idx = self.classes.index(label)
        dur = AudioUtils.get_duration(fp)
        
        if self.mode == "train":
            start = 0.0 if dur <= cfg.SEG_SEC else float(np.random.uniform(0, dur - cfg.SEG_SEC))
        else:
            start = 0.0 if dur <= cfg.SEG_SEC else max(0.0, (dur - cfg.SEG_SEC) / 2.0)
        
        waveform = AudioUtils.ffmpeg_read(fp, sr=cfg.SR, offset=start, duration=cfg.SEG_SEC)
        # Use comprehensive feature extraction instead of just log mel
        features = AudioUtils.extract_comprehensive_features(waveform, sr=cfg.SR)
        features = AudioUtils.fix_length_multi_channel(features)
        
        # shape: (n_channels, n_mels, frames) - no need to unsqueeze since we already have multiple channels
        tensor = torch.from_numpy(features)
        return tensor, label_idx

# ----------------------------- Model --------------------------------

class ResNetModel(nn.Module):
    """ResNet-based model for artist classification (initialized from scratch)."""
    
    def __init__(self, num_classes: int, input_channels: int = 11):
        super().__init__()
        # Create ResNet50 without pretrained weights (compat for different torchvision versions)
        try:
            # newer torchvision: weights=None disables pretrained weights
            self.backbone = models.resnet50(weights=None)
        except TypeError:
            # older torchvision: use pretrained=False
            self.backbone = models.resnet50(pretrained=False)
        
        # adapt to multi-channel input (11 channels: logmel + 10 additional feature channels)
        self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        infeat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(infeat, num_classes)
        
        # initialize weights for training from scratch
        self._init_weights()
    
    def _init_weights(self):
        """Custom weight initialization for training from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# --------------------- Label smoothing loss -------------------------

class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, eps: float = 0.05):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = logits.size(1)
        logp = F.log_softmax(logits, dim=1)
        # Negative log likelihood with smoothing
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logp.mean(dim=1)
        loss = (1.0 - self.eps) * nll + self.eps * smooth_loss / n_class
        return loss.mean()

# --------------------------- Inference helpers --------------------------

def segment_offsets_for_eval(duration: float, seg_s: float = cfg.SEG_SEC,
                           n: int = cfg.EVAL_SEGMENTS) -> List[float]:
    """Generate offset times for evaluation segments."""
    if duration <= seg_s:
        return [0.0]
    return list(np.linspace(0.0, duration - seg_s, n))

def predict_top3_for_audio(model: nn.Module, file_path: str, classes: List[str]) -> List[str]:
    """Predict top-3 artists for a single audio file."""
    dur = AudioUtils.get_duration(file_path)
    probs: List[np.ndarray] = []
    model.eval()
    
    with torch.no_grad():
        for start_offset in segment_offsets_for_eval(dur, cfg.SEG_SEC, cfg.EVAL_SEGMENTS):
            waveform = AudioUtils.ffmpeg_read(file_path, sr=cfg.SR, offset=start_offset, duration=cfg.SEG_SEC)
            # Use comprehensive feature extraction instead of just log mel
            features = AudioUtils.extract_comprehensive_features(waveform, sr=cfg.SR)
            features = AudioUtils.fix_length_multi_channel(features)
            x = torch.from_numpy(features).unsqueeze(0).to(DEVICE)  # (1, C, H, T)
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs.append(prob)

    # Average probabilities across segments
    prob_array = np.clip(np.array(probs), 1e-12, 1.0)
    # Geometric mean
    geo_mean = np.exp(np.log(prob_array).mean(axis=0))
    geo_mean = geo_mean / geo_mean.sum()
    
    # Get top-3 indices
    top_indices = np.argsort(geo_mean)[::-1][:3]
    return [classes[i] for i in top_indices]

# ------------------------- CPU info ---------------------------

def get_cpu_info():
    """Get CPU information for logging."""
    return {
        "cpu_count": _cpu_count,
        "workers": _workers,
        "device": DEVICE
    }
