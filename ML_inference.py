import os
import json
import warnings
import subprocess
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import multiprocessing

warnings.filterwarnings("ignore")
os.environ.setdefault("AUDIOREAD_BACKEND", "ffmpeg")

import numpy as np
import librosa
from tqdm import tqdm
from sklearn.preprocessing import normalize
import joblib


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    val_segment_duration: float = 8.0
    val_segments_per_track: int = 20
    random_seed: int = 42


@dataclass
class PathConfig:
    data_root: str = "./artist20"
    test_directory: str = field(init=False)
    output_predictions: str = "test_pred_ml.json"
    model_checkpoint: str = "checkpoint_ml.pkl"
    scaler_checkpoint: str = "scaler_ml.pkl"

    def __post_init__(self):
        self.test_directory = os.path.join(self.data_root, "test")


class FFmpegAudioLoader:
    @staticmethod
    def is_available() -> bool:
        return (
            shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
        )

    @staticmethod
    def load_audio(
        filepath: str,
        sample_rate: int,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = temp_wav.name
        temp_wav.close()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if offset > 0:
            command += ["-ss", f"{float(offset):.6f}"]
        command += ["-i", filepath]
        if duration and duration > 0:
            command += ["-t", f"{float(duration):.6f}"]
        command += ["-ac", "1", "-ar", str(sample_rate), temp_path, "-y"]
        subprocess.run(command, check=True)
        audio_signal, _ = librosa.load(temp_path, sr=sample_rate, mono=True)
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return audio_signal, sample_rate

    @staticmethod
    def get_duration(filepath: str, sample_rate: int) -> float:
        if FFmpegAudioLoader.is_available():
            try:
                probe_output = (
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "default=noprint_wrappers=1:nokey=1",
                            filepath,
                        ],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                duration = float(probe_output)
                if duration > 0:
                    return duration
            except Exception:
                pass
        audio_signal, _ = FFmpegAudioLoader.load_audio(filepath, sample_rate)
        return len(audio_signal) / float(sample_rate)


class AudioFeatureExtractor:
    def __init__(self, config: AudioConfig):
        self.cfg = config

    @staticmethod
    def compute_statistics(feature_matrix: np.ndarray) -> np.ndarray:
        mean_values = feature_matrix.mean(axis=1)
        std_values = feature_matrix.std(axis=1)
        return np.concatenate([mean_values, std_values], axis=0).astype(np.float32)

    def extract(
        self, filepath: str, start_time: float, segment_duration: float
    ) -> np.ndarray:
        audio_signal, sr = FFmpegAudioLoader.load_audio(
            filepath,
            self.cfg.sample_rate,
            offset=float(max(0.0, start_time)),
            duration=float(segment_duration),
        )
        if audio_signal.size == 0:
            raise RuntimeError("Empty audio file")
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_signal,
            sr=sr,
            n_mels=self.cfg.n_mels,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_spectrogram + 1e-10)
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=self.cfg.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        feature_vector = [
            self.compute_statistics(mfcc),
            self.compute_statistics(mfcc_delta),
            self.compute_statistics(mfcc_delta2),
        ]
        chroma = librosa.feature.chroma_stft(
            y=audio_signal, sr=sr, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        feature_vector.append(self.compute_statistics(chroma))
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio_signal, sr=sr, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )
        feature_vector.append(self.compute_statistics(spectral_contrast))
        try:
            harmonic_signal = librosa.effects.harmonic(audio_signal)
            tonnetz = librosa.feature.tonnetz(y=harmonic_signal, sr=sr)
            feature_vector.append(self.compute_statistics(tonnetz))
        except Exception:
            feature_vector.append(np.zeros(12, dtype=np.float32))
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_signal, sr=sr, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_signal, sr=sr, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_signal, sr=sr, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]
        spectral_flatness = librosa.feature.spectral_flatness(
            y=audio_signal, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]
        zero_crossing = librosa.feature.zero_crossing_rate(
            y=audio_signal, frame_length=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]
        rms_energy = librosa.feature.rms(
            y=audio_signal, frame_length=self.cfg.n_fft, hop_length=self.cfg.hop_length
        )[0]

        def mean_std(values):
            return np.array([values.mean(), values.std()], dtype=np.float32)

        feature_vector.extend(
            [
                mean_std(spectral_centroid),
                mean_std(spectral_bandwidth),
                mean_std(spectral_rolloff),
                mean_std(spectral_flatness),
                mean_std(zero_crossing),
                mean_std(rms_energy),
            ]
        )
        combined_features = np.concatenate(feature_vector, axis=0)
        combined_features = np.nan_to_num(
            combined_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        return combined_features


class SegmentSampler:
    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def sample_evaluation_offsets(
        self, total_duration: float, segment_duration: float, num_segments: int
    ) -> List[float]:
        if total_duration <= segment_duration:
            return [0.0]
        return list(np.linspace(0.0, total_duration - segment_duration, num_segments))


def list_audio_files(directory: str) -> List[str]:
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            if fname.lower().endswith(exts):
                files.append(os.path.join(root, fname))
    return sorted(files)


def main():
    audio_cfg = AudioConfig()
    path_cfg = PathConfig()
    cpu_count = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using n_jobs={cpu_count} for parallel processing (if applicable).")
    # Load model and scaler
    model = joblib.load(path_cfg.model_checkpoint)
    scaler = joblib.load(path_cfg.scaler_checkpoint)
    class_labels = list(model.classes_)
    feature_extractor = AudioFeatureExtractor(audio_cfg)
    sampler = SegmentSampler(audio_cfg.random_seed)
    test_files = list_audio_files(path_cfg.test_directory)
    print(f"Found {len(test_files)} test files.")
    results = {}
    for audio_file in tqdm(test_files, desc="Inference", ncols=100):
        duration = FFmpegAudioLoader.get_duration(audio_file, audio_cfg.sample_rate)
        offsets = sampler.sample_evaluation_offsets(
            duration,
            audio_cfg.val_segment_duration,
            audio_cfg.val_segments_per_track,
        )
        segment_features = [
            feature_extractor.extract(
                audio_file, offset, audio_cfg.val_segment_duration
            )
            for offset in offsets
        ]
        normalized_features = normalize(np.vstack(segment_features))
        scaled_segments = scaler.transform(normalized_features)
        segment_probabilities = model.predict_proba(scaled_segments)
        log_probabilities = np.log(np.clip(segment_probabilities, 1e-12, 1.0))
        averaged_proba = np.exp(log_probabilities.mean(axis=0))
        averaged_proba = averaged_proba / averaged_proba.sum()

        top3_idx = np.argsort(averaged_proba)[-3:][::-1]
        top3_labels = [class_labels[int(i)] for i in top3_idx]

        results[Path(audio_file).stem] = top3_labels

    with open(path_cfg.output_predictions, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved inference results to {path_cfg.output_predictions}")


if __name__ == "__main__":
    main()
