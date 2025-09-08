#!/usr/bin/env python3
"""Evaluation & Robustness Analysis of RIS-based Jamming ML Anomaly Detectors"""
"""Author: Hui Shing"""

import argparse
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_handler import (
    DatasetPaths, load_features, extract_feature_matrix,
    prepare_binary_classification
)
from utils import set_random_seeds, setup_logging, make_json_serialisable
from metrics import analyse_band_performance

warnings.filterwarnings('ignore')

def get_model_scores(model, X_test: np.ndarray, model_type: str) -> np.ndarray:
    """Get prediction scores from model (higher = more likely attack)."""
    if model_type == 'supervised':
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)[:, 1]
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test)
            return 1.0 / (1.0 + np.exp(-scores))  # sigmoid
        return model.predict(X_test).astype(float)

    # Unsupervised: higher = more anomalous
    if hasattr(model, 'decision_function'):
        return -model.decision_function(X_test)
    if hasattr(model, 'score_samples'):
        return -model.score_samples(X_test)
    raise ValueError("Unsupervised model lacks scoring method")

def load_model_artifacts(model_path: Path, use_stealthy: bool = False) -> Tuple[Any, float, List[str], str, str]:
    """Load model artifacts produced from detection.py."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger = logging.getLogger(__name__)
    bundle = joblib.load(model_path)

    model = bundle['model']
    model_type = bundle['model_type']
    champion_name = bundle['champion_name']
    features = list(bundle['final_features'])

    # Extract threshold based on mode
    thresholds = bundle['thresholds']
    if use_stealthy:
        threshold = float(thresholds['stealthy'])
        mode = "stealthy"
    else:
        threshold = float(thresholds['standard'])
        mode = "standard"

    # Log metadata
    logger.info("--------------------------------------------------")
    logger.info("Model artifacts")
    logger.info(f"  Path: {model_path}")
    logger.info(f"  Champion: {champion_name}")
    logger.info(f"  Type: {model_type}")
    logger.info(f"  Threshold ({mode}): {threshold:.3f}")
    logger.info(f"  Final features ({len(features)}):")
    for f in features:
        logger.info(f"    - {f}")
    logger.info("--------------------------------------------------")

    return model, threshold, features, model_type, champion_name

def load_test_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate test dataset."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'label' not in df.columns:
        raise ValueError("Test dataset must contain 'label' column")
    
    # Fill missing band_name values
    if 'band_name' in df.columns:
        df['band_name'] = df['band_name'].fillna('none')
    
    # Convert to binary if needed
    unique_labels = sorted(df['label'].unique())
    if len(unique_labels) > 2:
        logging.info(f"Converting multiclass to binary: {unique_labels} → [0,1]")
        df = prepare_binary_classification(df, normal_class=0, attack_class=1)
    
    return df

def evaluate_model_performance(model, threshold: float, X_test: np.ndarray, y_test: np.ndarray, model_type: str) -> Dict:
    """Evaluate model with comprehensive metrics."""
    
    # Get scores and predictions
    y_scores = get_model_scores(model, X_test, model_type)
    y_pred = (y_scores >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    # Basic metrics
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr = recall
    
    # Enhanced metrics
    sensitivity = tpr
    specificity = tnr
    balanced_accuracy = (sensitivity + specificity) / 2.0
    
    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    precision_normal, precision_ris = precisions
    recall_normal, recall_ris = recalls
    f1_normal, f1_ris = f1s
    support_normal, support_ris = supports

    macro_f1 = (f1_normal + f1_ris) / 2.0
    weighted_f1 = (
        (f1_normal * support_normal + f1_ris * support_ris) / (support_normal + support_ris)
        if (support_normal + support_ris) > 0 else 0.0
    )
    
    # AUC metrics
    try:
        if model_type == 'unsupervised':
            y_scores_norm = (y_scores - np.min(y_scores)) / (np.max(y_scores) - np.min(y_scores) + 1e-8)
            roc_auc = roc_auc_score(y_test, y_scores_norm) if len(np.unique(y_test)) > 1 else 0.0
            pr_auc = average_precision_score(y_test, y_scores_norm) if len(np.unique(y_test)) > 1 else 0.0
        else:
            roc_auc = roc_auc_score(y_test, y_scores) if len(np.unique(y_test)) > 1 else 0.0
            pr_auc = average_precision_score(y_test, y_scores) if len(np.unique(y_test)) > 1 else 0.0
    except Exception:
        roc_auc = 0.0
        pr_auc = 0.0
    
    return {
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'comprehensive_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'tpr': float(tpr),
            'tnr': float(tnr),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'balanced_accuracy': float(balanced_accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1_positive': float(f1_ris),
            'f1_negative': float(f1_normal),
            'precision_positive': float(precision_ris),
            'precision_negative': float(precision_normal),
            'recall_positive': float(recall_ris),
            'recall_negative': float(recall_normal),
        },
        'threshold_used': float(threshold)
    }

class ModelEvaluator:
    """Evaluator for trained models on test datasets."""
    
    def __init__(self, model, threshold: float, features: List[str], model_type: str, 
                 champion_name: str, threshold_mode: str, seed: int = 42):
        self.model = model
        self.threshold = threshold
        self.features = features
        self.model_type = model_type
        self.champion_name = champion_name
        self.threshold_mode = threshold_mode
        self.logger = logging.getLogger(__name__)
        self.seed = seed

    def evaluate_dataset(self, csv_path: Path) -> Dict:
        """Evaluate model on test dataset."""
        dataset_name = csv_path.stem
        self.logger.info(f"Evaluating on {dataset_name}")
        
        # Load test data
        df = load_test_dataset(csv_path)
        
        # Extract features
        X_test, y_test, available_features = extract_feature_matrix(df)
        
        # Check for missing features
        missing = [f for f in self.features if f not in available_features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Select features
        feature_indices = [available_features.index(f) for f in self.features]
        X_test_selected = X_test[:, feature_indices]
        
        performance = evaluate_model_performance(
            self.model, self.threshold, X_test_selected, y_test, self.model_type
        )
        
        # Band analysis
        y_pred = (get_model_scores(self.model, X_test_selected, self.model_type) >= self.threshold).astype(int)
        band_breakdown = analyse_band_performance(df, y_test, y_pred)
        
        # Calculate aggregate band metrics
        band_none_fpr = band_breakdown.get("none", {}).get("fpr")
        band_positive_fnr = None
        if band_breakdown:
            jam_bands = {k: v for k, v in band_breakdown.items() if k != "none"}
            if jam_bands:
                fnrs = [v.get("fnr") for v in jam_bands.values() if v.get("fnr") is not None]
                band_positive_fnr = max(fnrs) if fnrs else None
        
        results = {
            'dataset_name': dataset_name,
            'n_samples': len(y_test),
            'class_distribution': {'normal': int(np.sum(y_test == 0)), 'ris': int(np.sum(y_test == 1))},
            'features_used': self.features,
            'performance': {
                'threshold_used': self.threshold,
                'threshold_mode': self.threshold_mode,
                'confusion_matrix': performance['confusion_matrix'],
                'comprehensive_metrics': performance['comprehensive_metrics'],
                'band_breakdown': band_breakdown,
                'band_none_fpr': band_none_fpr,
                'band_positive_fnr': band_positive_fnr,
            }
        }
        
        cm = performance['confusion_matrix']
        metrics = performance['comprehensive_metrics']
        self.logger.info(f"Results for {dataset_name}:")
        self.logger.info(f"  Samples: {len(y_test)} (Normal: {np.sum(y_test==0)}, RIS: {np.sum(y_test==1)})")
        self.logger.info(f"  Confusion Matrix: TP={cm['tp']}, FP={cm['fp']}, TN={cm['tn']}, FN={cm['fn']}")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        self.logger.info(f"  FPR: {metrics['fpr']:.3f}, FNR: {metrics['fnr']:.3f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Evaluation and Robustness Analysis of ML models for RIS Jamming Detection')
    parser.add_argument('--model', required=True, help='Path to trained model file from detection.py')
    parser.add_argument('--test-csvs', nargs='+', required=True, help='Test dataset CSV files')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--threshold-mode', choices=['standard', 'stealthy'], default='standard', 
                       help='Which threshold to use from detection artifacts')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging('INFO', output_dir / 'robustness_evaluation.log')
    logger = logging.getLogger(__name__)
    
    set_random_seeds(args.seed)

    logger.info("Evaluation and Robustness Analysis")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test datasets: {len(args.test_csvs)}")
    logger.info(f"Threshold mode: {args.threshold_mode}")
    
    try:
        # Load model
        use_stealthy = (args.threshold_mode == 'stealthy')
        model, threshold, features, model_type, champion_name = load_model_artifacts(
            Path(args.model), use_stealthy=use_stealthy
        )
        
        evaluator = ModelEvaluator(
            model, threshold, features, model_type, champion_name, args.threshold_mode, args.seed
        )
        
        # Evaluate each test dataset
        all_results = []
        
        for test_csv in args.test_csvs:
            test_path = Path(test_csv)
            try:
                results = evaluator.evaluate_dataset(test_path)
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {test_path.name}: {e}")
        
        if not all_results:
            logger.error("No datasets evaluated successfully")
            return 1
        
        overall_summary = {
            'threshold_mode': args.threshold_mode,
            'threshold': threshold,
            'champion_name': champion_name,
            'model_type': model_type,
            'n_features': len(features),
            'final_features': features,
            'datasets_evaluated': [r['dataset_name'] for r in all_results],
            'evaluation_results': all_results
        }
        
        with open(output_dir / 'evaluation_complete_results.json', 'w') as f:
            json.dump(make_json_serialisable(overall_summary), f, indent=2)
        
        print(f"\nEvaluation and Robustness Analysis Results")
        print(f"Champion: {champion_name}")
        print(f"Threshold ({args.threshold_mode}): {threshold:.3f}")
        print(f"Datasets evaluated: {len(all_results)}")
        
        logger.info(f"Evaluation complete. Results saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation and analysis failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())