import time
from pathlib import Path
from typing import List
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from DL_utils import (
    Config,
    AudioDataset,
    ResNetModel,
    LabelSmoothingLoss,
    load_paths_from_json,
    label_from_filepath,
    get_cpu_info,
    DEVICE,
)
from DL_logger import logger

cfg = Config()


class Trainer:
    """Training class for the artist classification model."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        classes: List[str],
    ):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = classes

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.LR, weight_decay=1e-2
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
        )
        self.criterion = LabelSmoothingLoss(eps=0.05)

        self.best_top1 = 0.0

    def train_one_epoch(self, epoch_idx: int) -> float:
        """Train the model for one epoch."""
        self.model.train()
        losses = []
        epoch_start_time = time.time()

        loop = tqdm(
            self.train_loader, desc=f"Train Epoch {epoch_idx}/{cfg.EPOCHS}", leave=False
        )

        for batch_idx, (input_batch, target_batch) in enumerate(loop):
            input_batch = input_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_batch)
            loss = self.criterion(outputs, target_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        epoch_time = time.time() - epoch_start_time
        avg_loss = sum(losses) / len(losses) if losses else 0.0

        return avg_loss, epoch_time

    def evaluate(self):
        """Evaluate the model on validation set."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        loop = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for input_batch, target_batch in loop:
                input_batch = input_batch.to(DEVICE)
                outputs = self.model(input_batch)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()

                all_predictions.append(probabilities)
                all_targets.append(target_batch.numpy())

        if not all_predictions:
            return 0.0, 0.0

        # Compute metrics
        from sklearn.metrics import top_k_accuracy_score

        y_score = tuple(
            tuple(pred) for pred_set in all_predictions for pred in pred_set
        )
        y_true = tuple(tar for target_set in all_targets for tar in target_set)

        top1 = top_k_accuracy_score(
            y_true, y_score, k=1, labels=list(range(len(self.classes)))
        )
        top3 = top_k_accuracy_score(
            y_true, y_score, k=3, labels=list(range(len(self.classes)))
        )

        return float(top1), float(top3)

    def get_confusion_matrix(self):
        """Get confusion matrix for validation data."""
        self.model.eval()
        all_predictions = []
        all_targets = []

        loop = tqdm(self.val_loader, desc="Computing Confusion Matrix", leave=False)

        with torch.no_grad():
            for input_batch, target_batch in loop:
                input_batch = input_batch.to(DEVICE)
                outputs = self.model(input_batch)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                all_predictions.extend(predictions)
                all_targets.extend(target_batch.numpy())

        cm = confusion_matrix(
            all_targets, all_predictions, labels=list(range(len(self.classes)))
        )
        return cm, all_targets, all_predictions

    def save_confusion_matrix(self, cm, targets, predictions):
        """Save confusion matrix as visualization."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Plot confusion matrix
        im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Set labels
        ax.set(
            xticks=list(range(len(self.classes))),
            yticks=list(range(len(self.classes))),
            xticklabels=self.classes,
            yticklabels=self.classes,
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = ".2f"
        thresh = cm_normalized.max() / 2.0
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm_normalized[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                )

        # Add title and layout
        ax.set_title("Validation Confusion Matrix (Normalized)", fontsize=16, pad=20)
        plt.tight_layout()

        # Save the plot
        confusion_matrix_path = cfg.LOG_DIR / "confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to: {confusion_matrix_path}")

        # Also save raw confusion matrix data
        cm_data = {
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_normalized.tolist(),
            "class_names": self.classes,
            "total_samples": len(targets),
            "accuracy_by_class": {},
        }

        # Calculate per-class accuracy
        for i, class_name in enumerate(self.classes):
            class_mask = np.array(targets) == i
            if np.sum(class_mask) > 0:
                class_accuracy = cm[i, i] / float(np.sum(class_mask))
                cm_data["accuracy_by_class"][class_name] = class_accuracy

        cm_json_path = cfg.LOG_DIR / "confusion_matrix_data.json"
        with open(cm_json_path, "w", encoding="utf-8") as f:
            json.dump(cm_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Confusion matrix data saved to: {cm_json_path}")

        # Log summary statistics
        overall_accuracy = np.trace(cm) / float(np.sum(cm))
        logger.info(f"Overall validation accuracy: {overall_accuracy:.4f}")

        return confusion_matrix_path, cm_json_path

    def fit(self):
        """Main training loop."""
        logger.log_training_start(
            {
                "Batch Size": cfg.BATCH,
                "Epochs": cfg.EPOCHS,
                "Learning Rate": cfg.LR,
                "Optimizer": "AdamW",
                "Scheduler": "CosineAnnealingLR",
                "Loss Function": "LabelSmoothingLoss",
            }
        )

        for epoch in range(1, cfg.EPOCHS + 1):
            start_time = time.time()

            # Training
            train_loss, epoch_time = self.train_one_epoch(epoch)
            self.scheduler.step()

            # Validation
            val_top1, val_top3 = self.evaluate()

            # Check if best model
            is_best = val_top1 > self.best_top1
            if is_best:
                self.best_top1 = val_top1
                torch.save(self.model.state_dict(), cfg.BEST_MODEL)

            # Log progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed_time = time.time() - start_time

            logger.log_epoch_progress(
                epoch=epoch,
                total_epochs=cfg.EPOCHS,
                train_loss=train_loss,
                val_top1=val_top1,
                val_top3=val_top3,
                lr=current_lr,
                epoch_time=elapsed_time,
                is_best=is_best,
            )

        logger.log_training_complete()


def main():
    """Main training function."""
    # Log system information
    cpu_info = get_cpu_info()
    logger.log_system_info(cpu_info)

    # Load training and validation datasets
    try:
        train_paths = load_paths_from_json(cfg.TRAIN_JSON)
        val_paths = load_paths_from_json(cfg.VAL_JSON)

        # Extract class information
        classes = sorted(list({label_from_filepath(path) for path in train_paths}))

        logger.info(
            f"Loaded {len(train_paths)} training files and {len(val_paths)} validation files"
        )

    except Exception as e:
        logger.error(f"Failed to load dataset files: {e}")
        raise

    # Log dataset information
    logger.log_dataset_info(
        train_count=len(train_paths),
        val_count=len(val_paths),
        num_classes=len(classes),
        class_names=classes,
    )

    # Create datasets
    train_dataset = AudioDataset(train_paths, classes, mode="train")
    val_dataset = AudioDataset(val_paths, classes, mode="val")

    # Log model information
    model = ResNetModel(num_classes=len(classes))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log_model_info(
        model_params=total_params,
        model_size=f"{total_params / 1e6:.2f}M parameters ({trainable_params / 1e6:.2f}M trainable)",
    )

    # Create data loaders
    pin_memory = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH,
        shuffle=True,
        num_workers=cpu_info["workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH,
        shuffle=False,
        num_workers=cpu_info["workers"],
        pin_memory=pin_memory,
    )

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, classes)

    # Start training
    trainer.fit()

    # Generate confusion matrix after training completion
    logger.info("Generating confusion matrix...")
    cm, targets, predictions = trainer.get_confusion_matrix()
    confusion_matrix_path, cm_data_path = trainer.save_confusion_matrix(
        cm, targets, predictions
    )

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
