"""
Data loading and dataset manipulation utilities for RIS jamming detection.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetPaths:
    """Immutable data structure for dataset file paths."""
    csv_path: Path
    results_dir: Path
    
    def __post_init__(self):
        """Validate paths exist."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.results_dir.mkdir(parents=True, exist_ok=True)


def load_features(paths: DatasetPaths) -> pd.DataFrame:
    """
    Load feature dataset from CSV with validation.
    
    Args:
        paths: Dataset paths containing CSV file path
        
    Returns:
        DataFrame with loaded features and labels
    """
    logger.info(f"Loading features from {paths.csv_path}")
    
    try:
        df = pd.read_csv(paths.csv_path)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load CSV: {e}")
    
    if 'label' not in df.columns:
        raise ValueError("CSV must contain required 'label' column")
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def create_stratified_composite_key(df: pd.DataFrame, stratify_cols: List[str] = None) -> pd.Series:
    """
    Create composite stratification key from multiple columns.
    
    Args:
        df: Input DataFrame
        stratify_cols: Columns to use for stratification (default: ['label', 'band_name'])
        
    Returns:
        Series with composite stratification keys
    """
    if stratify_cols is None:
        stratify_cols = ['label']
        if 'band_name' in df.columns:
            stratify_cols.append('band_name')
    
    # Handle missing values
    df_strat = df[stratify_cols].copy()
    for col in stratify_cols:
        if col in df_strat.columns:
            df_strat[col] = df_strat[col].fillna('unknown')
    
    # Create composite key
    composite_key = df_strat[stratify_cols[0]].astype(str)
    for col in stratify_cols[1:]:
        if col in df_strat.columns:
            composite_key = composite_key + "_" + df_strat[col].astype(str)
    
    logger.info(f"Created composite stratification key from {stratify_cols}")
    logger.info(f"Stratification groups: {sorted(composite_key.unique())}")
    
    return composite_key


def split_train_val_test_stratified(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15,
    stratify_cols: List[str] = None, seed: int = 42, sort_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create robust stratified train/validation/test splits with deterministic ordering.
    
    Args:
        df: Input DataFrame with 'label' column
        test_size: Fraction for test set
        val_size: Fraction for validation set
        stratify_cols: Columns for composite stratification
        seed: Random seed for reproducibility
        sort_col: Column to sort by for deterministic ordering (uses all columns if None)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'label' column for stratification")
    
    # Ensure deterministic ordering for seed consistency
    df_sorted = df.copy()
    if sort_col and sort_col in df.columns:
        df_sorted = df_sorted.sort_values(sort_col).reset_index(drop=True)
    else:
        # Sort by all columns for complete determinism
        df_sorted = df_sorted.sort_values(list(df.columns)).reset_index(drop=True)
    
    # Create composite stratification key
    composite_key = create_stratified_composite_key(df_sorted, stratify_cols)
    
    # Check minimum samples per stratum
    stratum_counts = composite_key.value_counts()
    min_stratum_size = stratum_counts.min()
    
    if min_stratum_size < 3:
        logger.warning(f"Small strata detected (min size: {min_stratum_size}). "
                      "Falling back to label-only stratification")
        composite_key = df_sorted['label'].astype(str)
        stratum_counts = composite_key.value_counts()
        min_stratum_size = stratum_counts.min()
    
    if min_stratum_size < 2:
        raise ValueError(f"Insufficient samples for stratification (min: {min_stratum_size})")
    
    # First split: separate test set
    df_trainval, df_test = train_test_split(
        df_sorted,
        test_size=test_size,
        stratify=composite_key,
        random_state=seed
    )
    
    # Create stratification key for remaining data
    trainval_key = create_stratified_composite_key(df_trainval, stratify_cols)
    
    # Second split: separate train and validation
    relative_val_size = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=relative_val_size,
        stratify=trainval_key,
        random_state=seed
    )
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    logger.info(f"Stratified split complete - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Log stratification effectiveness
    for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        split_key = create_stratified_composite_key(split_df, stratify_cols)
        dist = split_key.value_counts().to_dict()
        logger.info(f"{split_name} stratification: {dist}")
    
    return df_train, df_val, df_test


def prepare_binary_classification(df: pd.DataFrame, normal_class: int = 0, 
                                 attack_class: int = 1) -> pd.DataFrame:
    """Prepare dataset for binary classification by filtering classes."""
    
    available_classes = set(df['label'].unique())
    required_classes = {normal_class, attack_class}
    
    if not required_classes.issubset(available_classes):
        raise ValueError(f"Required classes {sorted(required_classes)} not found. "
                        f"Available: {sorted(available_classes)}")
    
    # Filter to binary classes only
    binary_mask = df['label'].isin([normal_class, attack_class])
    binary_df = df[binary_mask].copy().reset_index(drop=True)
    
    # Remap labels to 0/1
    label_mapping = {normal_class: 0, attack_class: 1}
    binary_df['label'] = binary_df['label'].map(label_mapping)
    
    logger.info(f"Binary dataset prepared: {len(binary_df)} samples")
    logger.info(f"Class distribution: {binary_df['label'].value_counts().to_dict()}")
    
    return binary_df

def extract_feature_matrix(df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract feature matrix and labels from dataframe.
    Args:
        df: Input dataframe
        exclude_cols: Columns to exclude from features
    Returns: Tuple of (feature_matrix, labels, feature_names)
    """
    if exclude_cols is None:
        exclude_cols = ['label', 'band_name']
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        raise ValueError("No feature columns found after exclusions")
    
    # Extract numeric features only
    feature_df = df[feature_cols].select_dtypes(include=[np.number])
    
    if len(feature_df.columns) < len(feature_cols):
        dropped = set(feature_cols) - set(feature_df.columns)
        logger.warning(f"Dropped non-numeric feature columns: {dropped}")
    
    X = feature_df.values
    y = df['label'].values
    feature_names = list(feature_df.columns)
    
    # Handle missing values
    if np.any(pd.isnull(X)):
        logger.warning("Found NaN values in features, filling with column medians")
        X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
        
    return X, y, feature_names
