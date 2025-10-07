import os
import json
import time
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
)
import joblib
from sklearn.feature_selection import f_classif


@dataclass
class AudioConfig:
    """Audio processing configuration"""

    sample_rate: int = 16000
    n_mels: int = 128
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    train_segment_duration: float = 8.0
    val_segment_duration: float = 8.0
    train_segments_per_track: int = 5
    val_segments_per_track: int = 20
    random_seed: int = 42


@dataclass
class PathConfig:
    """Path configuration"""

    data_root: str = "./artist20"
    train_json: str = field(init=False)
    val_json: str = field(init=False)
    test_directory: str = field(init=False)
    output_predictions: str = "test_pred_ml.json"
    model_checkpoint: str = "checkpoint_ml.pkl"
    scaler_checkpoint: str = "scaler_ml.pkl"

    def __post_init__(self):
        self.train_json = os.path.join(self.data_root, "train.json")
        self.val_json = os.path.join(self.data_root, "val.json")
        self.test_directory = os.path.join(self.data_root, "test")


class FFmpegAudioLoader:
    """Utility class for loading audio using FFmpeg"""

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
    """Audio feature extractor"""

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
    """Audio segment sampler"""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def sample_training_offsets(
        self, total_duration: float, segment_duration: float, num_segments: int
    ) -> List[float]:
        if total_duration <= segment_duration:
            return [0.0]
        rng = np.random.RandomState(self.seed + int(total_duration * 1000) % 9973)
        offsets = rng.uniform(
            0.0, max(1e-6, total_duration - segment_duration), size=num_segments
        )
        return list(np.sort(offsets))

    def sample_evaluation_offsets(
        self, total_duration: float, segment_duration: float, num_segments: int
    ) -> List[float]:
        if total_duration <= segment_duration:
            return [0.0]
        return list(np.linspace(0.0, total_duration - segment_duration, num_segments))


class DatasetBuilder:
    """Dataset builder"""

    def __init__(self, audio_config: AudioConfig):
        self.audio_cfg = audio_config
        self.feature_extractor = AudioFeatureExtractor(audio_config)
        self.sampler = SegmentSampler(audio_config.random_seed)

    @staticmethod
    def load_file_list(json_path: str) -> List[str]:
        base_directory = os.path.dirname(os.path.abspath(json_path))
        with open(json_path, "r", encoding="utf-8") as file:
            relative_paths = json.load(file)
        absolute_paths = []
        for path in relative_paths:
            path = os.path.normpath(path)
            if path.startswith("." + os.sep):
                path = path[2:]
            full_path = (
                path
                if os.path.isabs(path)
                else os.path.normpath(os.path.join(base_directory, path))
            )
            absolute_paths.append(full_path)
        return absolute_paths

    @staticmethod
    def extract_artist_label(filepath: str) -> str:
        path_components = os.path.normpath(filepath).split(os.sep)
        if "train_val" in path_components:
            index = path_components.index("train_val")
            if index + 1 < len(path_components):
                return path_components[index + 1]
        return path_components[-2]

    def build_training_dataset(
        self, file_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        features_list = []
        labels_list = []
        progress_bar = tqdm(
            file_list,
            desc="Extracting training features",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
        for audio_file in progress_bar:
            if not os.path.isfile(audio_file):
                continue
            duration = FFmpegAudioLoader.get_duration(
                audio_file, self.audio_cfg.sample_rate
            )
            offsets = self.sampler.sample_training_offsets(
                duration,
                self.audio_cfg.train_segment_duration,
                self.audio_cfg.train_segments_per_track,
            )
            for offset in offsets:
                feature_vec = self.feature_extractor.extract(
                    audio_file, offset, self.audio_cfg.train_segment_duration
                )
                features_list.append(feature_vec)
                labels_list.append(self.extract_artist_label(audio_file))
        feature_matrix = np.vstack(features_list)
        label_array = np.array(labels_list)
        feature_matrix = normalize(feature_matrix)
        return feature_matrix, label_array

    def build_evaluation_tracks(
        self, file_list: List[str]
    ) -> List[Tuple[np.ndarray, str, str]]:
        track_data = []
        progress_bar = tqdm(
            file_list,
            desc="Extracting evaluation features",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
        for audio_file in progress_bar:
            if not os.path.isfile(audio_file):
                continue
            duration = FFmpegAudioLoader.get_duration(
                audio_file, self.audio_cfg.sample_rate
            )
            offsets = self.sampler.sample_evaluation_offsets(
                duration,
                self.audio_cfg.val_segment_duration,
                self.audio_cfg.val_segments_per_track,
            )
            segment_features = [
                self.feature_extractor.extract(
                    audio_file, offset, self.audio_cfg.val_segment_duration
                )
                for offset in offsets
            ]
            normalized_features = normalize(np.vstack(segment_features))
            artist_label = self.extract_artist_label(audio_file)
            track_data.append((normalized_features, artist_label, audio_file))
        return track_data


class SVMClassifierTrainer:
    """SVM classifier trainer"""

    def __init__(self, random_seed: int = 42, n_jobs: int = 1):
        self.seed = random_seed
        self.model = None
        self.scaler = None
        self.class_labels = None
        self.n_jobs = n_jobs

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        print("\n" + "=" * 60)
        print("Standardizing features and starting grid search...")
        print("=" * 60)
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        hyperparameter_grid = {
            "C": [1, 2, 5, 10, 20],
            "gamma": ["scale", 1e-3, 5e-4, 1e-4],
        }
        base_svm = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=self.seed,
        )
        cross_validator = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=self.seed
        )
        grid_search = GridSearchCV(
            base_svm,
            hyperparameter_grid,
            scoring="accuracy",
            cv=cross_validator,
            n_jobs=self.n_jobs,
            verbose=1,
        )
        grid_search.fit(scaled_features, labels)
        self.model = grid_search.best_estimator_
        self.class_labels = list(np.unique(labels))
        print(f"\nBest hyperparameters: {grid_search.best_params_}")

    def save_checkpoint(self, model_path: str, scaler_path: str) -> None:
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"\nModel saved:")
        print(f"   ├─ {model_path}")
        print(f"   └─ {scaler_path}")

    def predict_track(self, segment_features: np.ndarray) -> Tuple[str, np.ndarray]:
        scaled_segments = self.scaler.transform(segment_features)
        segment_probabilities = self.model.predict_proba(scaled_segments)
        model_class_order = list(self.model.classes_)
        reorder_indices = [model_class_order.index(cls) for cls in self.class_labels]
        log_probabilities = np.log(np.clip(segment_probabilities, 1e-12, 1.0))
        averaged_proba = np.exp(log_probabilities.mean(axis=0))[reorder_indices]
        averaged_proba = averaged_proba / averaged_proba.sum()
        predicted_label = self.class_labels[int(np.argmax(averaged_proba))]
        return predicted_label, averaged_proba


class ModelEvaluator:
    """Model evaluator"""

    def __init__(self, class_labels: List[str]):
        self.labels = class_labels

    def evaluate(
        self,
        classifier: SVMClassifierTrainer,
        track_data: List[Tuple[np.ndarray, str, str]],
    ) -> dict:
        true_labels = []
        predicted_labels = []
        probability_scores = []
        progress_bar = tqdm(
            track_data,
            desc="Inferring validation set",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )
        for segment_features, true_label, _ in progress_bar:
            pred_label, proba = classifier.predict_track(segment_features)
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
            probability_scores.append(proba)
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        probability_matrix = np.vstack(probability_scores)
        conf_matrix = confusion_matrix(
            true_labels, predicted_labels, labels=self.labels
        )
        top1_accuracy = top_k_accuracy_score(
            true_labels, probability_matrix, k=1, labels=self.labels
        )
        top3_accuracy = top_k_accuracy_score(
            true_labels, probability_matrix, k=3, labels=self.labels
        )
        return {
            "confusion_matrix": conf_matrix,
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "true_labels": true_labels,
            "predicted_labels": predicted_labels,
        }

    def print_results(self, evaluation_results: dict) -> None:
        print("\n" + "=" * 60)
        print("Validation set performance")
        print("=" * 60)
        print(f"Top-1 Accuracy: {evaluation_results['top1_accuracy']:.4f}")
        print(f"Top-3 Accuracy: {evaluation_results['top3_accuracy']:.4f}")
        print("\nDetailed classification report:")
        print(
            classification_report(
                evaluation_results["true_labels"],
                evaluation_results["predicted_labels"],
                labels=self.labels,
                zero_division=0,
            )
        )

    def save_confusion_matrices(self, confusion_matrix: np.ndarray) -> None:
        matrix_configs = [
            (
                confusion_matrix,
                "Validation Set Confusion Matrix",
                "val_confusion_matrix.png",
            ),
            (
                confusion_matrix.astype(float)
                / confusion_matrix.sum(axis=1, keepdims=True),
                "Validation Set Confusion Matrix (Row Normalized)",
                "val_confusion_matrix_norm.png",
            ),
        ]
        for matrix, title, filename in matrix_configs:
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                matrix,
                cmap="Blues",
                annot=True,
                fmt=".2f",
                xticklabels=self.labels,
                yticklabels=self.labels,
            )
            plt.title(title, fontsize=14, pad=20)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.ylabel("True Label", fontsize=12)
            plt.tight_layout()
            plt.savefig(filename, dpi=220)
            plt.close()
            print(f"Saved: {filename}")


class FeatureImportanceAnalyzer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _build_segment_dataset(
        self, val_tracks: List[Tuple[np.ndarray, str, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for segment_features, artist_label, _ in val_tracks:
            if segment_features.size == 0:
                continue
            X_list.append(segment_features)  # shape: (n_segments, n_features)
            y_list.extend([artist_label] * segment_features.shape[0])
        if len(X_list) == 0:
            return np.zeros((0, 0)), np.array([])
        X = np.vstack(X_list)
        y = np.array(y_list)
        return X, y

    def compute_univariate_importance(
        self,
        classifier: "SVMClassifierTrainer",
        val_tracks: List[Tuple[np.ndarray, str, str]],
    ) -> dict:
        X, y = self._build_segment_dataset(val_tracks)
        if X.size == 0:
            print("No validation segments available for feature importance.")
            return {}
        print("\nComputing univariate feature scores (ANOVA F-test)...")

        F_vals, p_vals = f_classif(X, y)
        means = F_vals.astype(float)
        stds = np.zeros_like(means)
        sorted_idx = np.argsort(means)[::-1]
        return {
            "means": means,
            "stds": stds,
            "sorted_idx": sorted_idx,
            "feature_count": X.shape[1],
            "n_samples": X.shape[0],
        }

    def save_and_report(
        self,
        importance_dict: dict,
        output_prefix: str = "feature_importance",
        top_k: int = 50,
    ):
        if not importance_dict:
            return
        means = importance_dict["means"]
        stds = importance_dict["stds"]
        sorted_idx = importance_dict["sorted_idx"]
        feature_count = int(importance_dict["feature_count"])
        top_k = min(top_k, feature_count)
        top_indices = sorted_idx[:top_k]

        out = {
            "feature_count": feature_count,
            "n_top": top_k,
            "means": [float(m) for m in means.tolist()],
            "stds": [float(s) for s in stds.tolist()],
            "top_indices": [int(i) for i in top_indices.tolist()],
        }
        json_path = f"{output_prefix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved feature importances JSON: {json_path}")

        try:
            indices = top_indices
            values = means[indices]
            errors = stds[indices]
            plt.figure(figsize=(min(20, top_k * 0.3 + 4), 6))
            x_labels = [f"f{int(i)}" for i in indices]
            plt.bar(
                range(len(indices)),
                values,
                yerr=errors,
                align="center",
                alpha=0.8,
                ecolor="black",
                capsize=3,
            )
            plt.xticks(range(len(indices)), x_labels, rotation=90)
            plt.ylabel("Univariate score (ANOVA F-value)")
            plt.xlabel("Feature index")
            plt.title("Top feature univariate scores (ANOVA F-test)")
            plt.tight_layout()
            png_path = f"{output_prefix}.png"
            plt.savefig(png_path, dpi=220)
            plt.close()
            print(f"Saved feature importance plot: {png_path}")
        except Exception as e:
            print(f"Failed to plot feature importances: {e}")


class Artist20Pipeline:
    """Full training pipeline"""

    def __init__(self, audio_config: AudioConfig, path_config: PathConfig, n_jobs: int):
        self.audio_cfg = audio_config
        self.path_cfg = path_config
        self.n_jobs = n_jobs
        np.random.seed(audio_config.random_seed)

    def run(self) -> None:
        start_time = time.time()
        print("\n" + "=" * 60)
        print("Artist20 Music Classification System")
        print("=" * 60)
        dataset_builder = DatasetBuilder(self.audio_cfg)
        train_files = dataset_builder.load_file_list(self.path_cfg.train_json)
        val_files = dataset_builder.load_file_list(self.path_cfg.val_json)
        print(f"\nDataset statistics:")
        print(f"   ├─ Training set: {len(train_files)} tracks")
        print(f"   └─ Validation set: {len(val_files)} tracks")
        train_features, train_labels = dataset_builder.build_training_dataset(
            train_files
        )
        val_tracks = dataset_builder.build_evaluation_tracks(val_files)
        classifier = SVMClassifierTrainer(
            self.audio_cfg.random_seed, n_jobs=self.n_jobs
        )
        classifier.train(train_features, train_labels)
        classifier.save_checkpoint(
            self.path_cfg.model_checkpoint, self.path_cfg.scaler_checkpoint
        )
        evaluator = ModelEvaluator(classifier.class_labels)
        results = evaluator.evaluate(classifier, val_tracks)
        evaluator.print_results(results)
        evaluator.save_confusion_matrices(results["confusion_matrix"])

        try:
            analyzer = FeatureImportanceAnalyzer(
                random_state=self.audio_cfg.random_seed,
            )
            importance = analyzer.compute_univariate_importance(classifier, val_tracks)
            analyzer.save_and_report(
                importance, output_prefix="feature_importances", top_k=50
            )
        except Exception as e:
            print(f"Feature importance analysis failed: {e}")

        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete! Total time: {elapsed_time:.1f} seconds")
        print("=" * 60 + "\n")


def main():
    audio_cfg = AudioConfig()
    path_cfg = PathConfig()
    cpu_count = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using n_jobs={cpu_count} for parallel processing.")
    pipeline = Artist20Pipeline(audio_cfg, path_cfg, n_jobs=cpu_count)
    pipeline.run()


if __name__ == "__main__":
    main()
