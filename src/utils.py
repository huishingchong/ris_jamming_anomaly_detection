"""
Utility functions - includes seeding, logging setup, etc.
"""
import os
import sys
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import numpy as np


def setup_logging(level: str = 'INFO', log_file: Optional[Path] = None, format_str: Optional[str] = None) -> None:
    """
    Setup logging configuration for the research pipeline.
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        format_str: Custom format string
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if requested
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")


def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set sklearn random state (if available)
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass
    
    logging.info(f"Set random seeds to {seed}")

def make_json_serialisable(obj):
    """Convert numpy types to native Python types for JSON serialisation"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: make_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serialisable(item) for item in obj]
    return obj

def validate_dataset_integrity(features: np.ndarray, labels: np.ndarray, feature_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Validate dataset integrity and identify potential issues.
    Args:
        features: Feature array
        labels: Label array
        feature_names: Optional feature names for reporting
    """
    report = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    if features.shape[0] != labels.shape[0]:
        report['errors'].append(f"Feature/label count mismatch: {features.shape[0]} vs {labels.shape[0]}")
        report['valid'] = False
    
    nan_features = np.isnan(features).sum()
    inf_features = np.isinf(features).sum()
    
    if nan_features > 0:
        report['warnings'].append(f"Found {nan_features} NaN values in features")
    
    if inf_features > 0:
        report['warnings'].append(f"Found {inf_features} infinite values in features")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_distribution = dict(zip(unique_labels.astype(int), counts.astype(int)))
    
    # Prevent dividiong by 0
    if len(counts) > 1:
        min_class_size = min(counts)
        max_class_size = max(counts)
        imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
        
        if imbalance_ratio > 10:
            report['warnings'].append(f"High class imbalance: ratio {imbalance_ratio:.1f}")
    else:
        imbalance_ratio = 1.0
    
    # Check for constant features
    feature_variances = np.var(features, axis=0)
    constant_features = np.sum(feature_variances < 1e-10)
    
    if constant_features > 0:
        report['warnings'].append(f"Found {constant_features} near-constant features")
    
    report['statistics'] = {
        'n_samples': features.shape[0],
        'n_features': features.shape[1],
        'label_distribution': label_distribution,
        'class_imbalance_ratio': float(imbalance_ratio),
        'nan_values': int(nan_features),
        'inf_values': int(inf_features),
        'constant_features': int(constant_features)
    }
    
    return report
