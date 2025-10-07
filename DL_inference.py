"""
DL Artist Classification Inference Pipeline

This script handles inference on test datasets using a trained artist classification model.
It generates top-3 predictions for each audio file and saves results to JSON format.
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from DL_utils import (
    Config, AudioDataset, ResNetModel, AudioUtils, load_paths_from_json,
    label_from_filepath, predict_top3_for_audio, DEVICE, get_cpu_info
)
from DL_logger import logger

cfg = Config()

class InferenceEngine:
    """Inference engine for artist classification."""
    
    def __init__(self, model_path: str, classes: List[str]):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to the trained model checkpoints
            classes: List of artist class names
        """
        self.classes = classes
        self.model = ResNetModel(num_classes=len(classes), input_channels=11).to(DEVICE)  # 11 channels for comprehensive features
        
        # Load model weights
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """Load trained model weights."""
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            state_dict = torch.load(model_file, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
            
    def predict_batch(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Predict top-3 artists for a batch of audio files.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            Dictionary mapping file basename to top-3 artist predictions
        """
        logger.log_inference_start(len(file_paths))
        
        predictions = {}
        successful_predictions = 0
        
        for i, file_path in enumerate(tqdm(file_paths, desc="Processing audio files")):
            try:
                # Extract filename without extension as key
                filename = Path(file_path).stem
                
                # Get predictions
                top3_predictions = predict_top3_for_audio(self.model, file_path, self.classes)
                predictions[filename] = top3_predictions
                successful_predictions += 1
                
                # Log progress every 10 files
                if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
                    logger.log_inference_progress(i + 1, len(file_paths), file_path)
                    
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue
                
        logger.log_inference_complete(cfg.TEST_OUT, successful_predictions, len(file_paths))
        return predictions
        
    def predict_single_file(self, file_path: str) -> List[str]:
        """
        Predict top-3 artists for a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            List of top-3 artist predictions
        """
        return predict_top3_for_audio(self.model, file_path, self.classes)

def load_test_files() -> List[str]:
    """Load test files from either JSON or directory."""
    test_files = []
    
    # Try loading from JSON first
    if cfg.TEST_JSON.exists():
        test_files = load_paths_from_json(cfg.TEST_JSON)
        logger.info(f"Loaded {len(test_files)} test files from JSON: {cfg.TEST_JSON}")
    # Fallback to directory
    elif cfg.TEST_DIR.exists():
        supported_extensions = (".mp3", ".wav", ".flac")
        test_dir = cfg.TEST_DIR
        
        for file_name in sorted(os.listdir(test_dir)):
            if file_name.lower().endswith(supported_extensions):
                test_files.append(str(test_dir / file_name))
                
        logger.info(f"Loaded {len(test_files)} test files from directory: {test_dir}")
    else:
        raise FileNotFoundError(
            f"Test data not found. Please ensure either {cfg.TEST_JSON} or {cfg.TEST_DIR} exists."
        )
    
    # Filter existing files only
    existing_files = [f for f in test_files if os.path.isfile(f)]
    if len(existing_files) != len(test_files):
        logger.warning(f"Some test files are missing: {len(test_files) - len(existing_files)} files")
        
    return existing_files

def extract_classes_from_json():
    """Extract class information from training/validation JSON files."""
    try:
        if cfg.TRAIN_JSON.exists():
            train_paths = load_paths_from_json(cfg.TRAIN_JSON)
            classes = sorted(list({label_from_filepath(p) for p in train_paths}))
        elif cfg.VAL_JSON.exists():
            val_paths = load_paths_from_json(cfg.VAL_JSON)
            classes = sorted(list({label_from_filepath(p) for p in val_paths}))
        else:
            raise FileNotFoundError("Cannot find train.json or val.json for class extraction")
            
        return classes
    except Exception as e:
        raise RuntimeError(f"Failed to extract classes: {e}")

def main():
    """Main inference function."""
    # Log system information
    cpu_info = get_cpu_info()
    logger.log_system_info(cpu_info)
    
    # Check if model exists
    if not Path(cfg.BEST_MODEL).exists():
        raise FileNotFoundError(
            f"Trained model not found: {cfg.BEST_MODEL}. "
            "Please run DL_train.py first to train the model."
        )
    
    # Extract class information
    try:
        classes = extract_classes_from_json()
        logger.log_dataset_info(
            train_count=0,  # Not relevant for inference
            val_count=0,    # Not relevant for inference  
            num_classes=len(classes),
            class_names=classes
        )
    except Exception as e:
        logger.error(f"Failed to extract class information: {e}")
        raise
    
    # Load test files
    try:
        test_files = load_test_files()
    except Exception as e:
        logger.error(f"Failed to load test files: {e}")
        raise
    
    if not test_files:
        logger.warning("No test files found. Exiting.")
        return
        
    # Initialize inference engine
    try:
        engine = InferenceEngine(cfg.BEST_MODEL, classes)
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    # Run inference
    logger.info("=" * 80)
    logger.info("RUNNING ARTIST CLASSIFICATION INFERENCE")
    logger.info("=" * 80)
    
    predictions = engine.predict_batch(test_files)
    
    # Save predictions
    output_file = cfg.TEST_OUT
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved successfully to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        raise
    
    # Print summary statistics
    logger.info("=" * 80)
    logger.info("INFERENCE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {len(predictions)}")
    logger.info(f"Output file: {output_file}")
    
    # Show prediction statistics
    prediction_counts = {}
    for top3 in predictions.values():
        top1 = top3[0]
        prediction_counts[top1] = prediction_counts.get(top1, 0) + 1
    
    logger.info("Top-1 prediction distribution:")
    for artist, count in sorted(prediction_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(predictions)) * 100
        logger.info(f"  {artist}: {count} files ({percentage:.1f}%)")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
