import logging
import logging.handlers
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

try:
    import torch
    import numpy as np

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DLMFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def __init__(self):
        super().__init__()

    def format(self, record):
        # Color codes for different log levels
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")

        # Format level name with color
        level_name = record.levelname
        if sys.stderr.isatty():  # Only use colors if terminal supports them
            colored_level = f"{colors.get(level_name, '')}{level_name}{reset}"
        else:
            colored_level = level_name

        # Format log name
        logger_name = record.name
        if logger_name == "root":
            logger_name = "DL"

        # Create formatted message
        formatted = f"{timestamp} | {colored_level:<8} | {logger_name:<15} | {record.getMessage()}"
        return formatted


class PerformanceMetrics:
    """Track training performance metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.epoch_data = []
        self.start_time = datetime.now()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_top1: float,
        val_top3: float,
        lr: float,
        epoch_time: float,
    ):
        """Log metrics for a training epoch."""
        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "learning_rate": lr,
            "epoch_time": epoch_time,
            "timestamp": datetime.now().isoformat(),
        }
        self.epoch_data.append(epoch_info)

    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics."""
        if not self.epoch_data:
            return {}

        best_epoch = max(self.epoch_data, key=lambda x: x["val_top1"])
        return {
            "best_epoch": best_epoch["epoch"],
            "best_top1": best_epoch["val_top1"],
            "best_top3": best_epoch["val_top3"],
            "train_loss_at_best": best_epoch["train_loss"],
        }

    def get_training_summary(self) -> Dict[str, Any]:
        """Get complete training summary."""
        if not self.epoch_data:
            return {}

        total_time = (datetime.now() - self.start_time).total_seconds()
        final_lr = self.epoch_data[-1]["learning_rate"]

        return {
            "total_epochs": len(self.epoch_data),
            "total_training_time": total_time,
            "final_learning_rate": final_lr,
            "metrics": self.get_best_metrics(),
            "epoch_data": self.epoch_data,
        }


class DLMLogger:
    """Main logger class for DL system."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.log_dir = Path("./logs")
        self.log_dir.mkdir(exist_ok=True)

        self.metrics = PerformanceMetrics()
        self.logger = self._setup_logger()
        self._initialized = True

    def _setup_logger(self) -> logging.Logger:
        """Setup the main logger with file and console handlers."""
        logger = logging.getLogger("DL")
        logger.setLevel(logging.INFO)

        # Clear existing handlers
        logger.handlers.clear()

        # Create custom formatter
        formatter = DLMFormatter()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler - daily rotation
        log_file = self.log_dir / "dl_training.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def log_system_info(self, cpu_info: Dict[str, Any]):
        """Log system configuration information."""
        self.logger.info("=" * 80)
        self.logger.info("DL ARTIST CLASSIFICATION SYSTEM INITIALIZATION")
        self.logger.info("=" * 80)
        self.logger.info(f"System Information:")
        self.logger.info(f"  CPU cores: {cpu_info['cpu_count']}")
        self.logger.info(f"  Worker threads: {cpu_info['workers']}")
        self.logger.info(f"  Device: {cpu_info['device']}")
        if TORCH_AVAILABLE:
            self.logger.info(f"  PyTorch version: {torch.__version__}")
            self.logger.info(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                self.logger.info(f"  CUDA version: {torch.version.cuda}")
                self.logger.info(f"  GPU name: {torch.cuda.get_device_name(0)}")
                self.logger.info(
                    f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
        else:
            self.logger.info("  PyTorch: Not available")
            self.logger.info("  CUDA: Not available")
        self.logger.info(f"  Log directory: {self.log_dir.resolve()}")
        self.logger.info("-" * 80)

    def log_dataset_info(
        self, train_count: int, val_count: int, num_classes: int, class_names: list
    ):
        """Log dataset information."""
        self.logger.info(f"Dataset Information:")
        self.logger.info(f"  Training samples: {train_count:,}")
        self.logger.info(f"  Validation samples: {val_count:,}")
        self.logger.info(f"  Number of classes: {num_classes}")
        self.logger.info(f"  Classes: {', '.join(class_names)}")
        self.logger.info("-" * 80)

    def log_model_info(self, model_params: int, model_size: str):
        """Log model architecture information."""
        self.logger.info(f"Model Information:")
        self.logger.info(f"  Parameters: {model_params:,}")
        self.logger.info(f"  Model size: {model_size}")
        self.logger.info(f"  Architecture: ResNet50 (adapted for audio)")
        self.logger.info("-" * 80)

    def log_training_start(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.logger.info("TRAINING CONFIGURATION:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 80)
        self.logger.info("STARTING TRAINING...")

    def log_epoch_progress(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_top1: float,
        val_top3: float,
        lr: float,
        epoch_time: float,
        is_best: bool = False,
    ):
        """Log epoch training progress."""
        status = "★ BEST" if is_best else ""
        self.logger.info(
            f"Epoch [{epoch:3d}/{total_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Top-1: {val_top1:.4f} | "
            f"Val Top-3: {val_top3:.4f} | "
            f"LR: {lr:.2e} | "
            f"Time: {epoch_time:.1f}s {status}"
        )

        # Log to metrics
        self.metrics.log_epoch(epoch, train_loss, val_top1, val_top3, lr, epoch_time)

    def log_inference_progress(self, processed: int, total: int, current_file: str):
        """Log inference progress."""
        if processed % 10 == 0 or processed == total:  # Log every 10 files
            percentage = (processed / total) * 100
            self.logger.info(
                f"Processing file {processed}/{total} ({percentage:.1f}%): {current_file}"
            )

    def log_training_complete(self):
        """Log training completion with best metrics."""
        summary = self.metrics.get_training_summary()
        if summary:
            self.logger.info("=" * 80)
            self.logger.info("TRAINING COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            metrics = summary["metrics"]
            self.logger.info(f"Best Results:")
            self.logger.info(f"  Epoch: {metrics['best_epoch']}")
            self.logger.info(f"  Top-1 Accuracy: {metrics['best_top1']:.4f}")
            self.logger.info(f"  Top-3 Accuracy: {metrics['best_top3']:.4f}")
            self.logger.info(f"  Training Loss: {metrics['train_loss_at_best']:.4f}")
            self.logger.info(
                f"  Total Training Time: {summary['total_training_time']:.1f}s"
            )
            self.logger.info(f"  Total Epochs: {summary['total_epochs']}")

            # Save detailed metrics to file
            metrics_file = self.log_dir / "training_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Detailed metrics saved to: {metrics_file}")

    def log_inference_start(self, num_files: int):
        """Log inference start."""
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING INFERENCE ON {num_files} FILES")
        self.logger.info("=" * 80)

    def log_inference_complete(
        self, output_file: str, success_count: int, total_count: int
    ):
        """Log inference completion."""
        self.logger.info("=" * 80)
        self.logger.info("INFERENCE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"Files processed: {success_count}/{total_count}")
        self.logger.info(f"Predictions saved to: {output_file}")

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)


# Global logger instance
logger = DLMLogger()
