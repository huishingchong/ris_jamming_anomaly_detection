"""
Machine learning model utility for RIS jamming detection.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.base import clone
from scipy.stats import loguniform, uniform, randint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a machine learning model with hyperparameters"""
    name: str
    pipeline: Pipeline
    param_grid: Dict[str, List[Any]]
    description: str = ""


@dataclass
class ModelResult:
    """Results from model training and evaluation"""
    name: str
    best_params: Dict[str, Any]
    best_score: float
    metrics: Dict[str, float]
    threshold: float
    model: Pipeline = field(repr=False)
    latency_ms: Optional[float] = None


def get_supervised_candidates() -> List[ModelSpec]:
    """
    Get supervised model candidates for RQ1 analysis.
    Returns: List of ModelSpec objects for supervised learning
    """
    # Logistic Regression with comprehensive regularisation
    logreg_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000, random_state=42))
    ])
    
    logreg_param_grid = [
        {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'classifier__class_weight': ['balanced', None]
        },
        {
            'classifier__C': [0.01, 0.1, 1.0, 10.0],
            'classifier__penalty': ['l1'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__class_weight': ['balanced', None]
        },
        {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['elasticnet'],
            'classifier__solver': ['saga'],
            'classifier__l1_ratio': [0.1, 0.5, 0.7, 0.9],
            'classifier__class_weight': ['balanced', None]
        }
    ]
    
    # SVM with multiple kernels
    svm_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    svm_param_grid = [
        {
            'classifier__kernel': ['rbf'],
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'classifier__class_weight': ['balanced', None]
        },
        {
            'classifier__kernel': ['linear'],
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__class_weight': ['balanced', None]
        },
        {
            'classifier__kernel': ['poly'],
            'classifier__degree': [2, 3, 4],
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__class_weight': ['balanced', None]
        }
    ]
    
    # Random Forest
    rf_pipe = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    rf_param_grid = {
        'classifier__n_estimators': [200, 300, 500, 800],
        'classifier__max_depth': [None, 20, 30, 50],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2', 0.5],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Gradient Boosting
    gb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    
    gb_param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__subsample': [0.8, 0.9, 1.0]
    }
    
    candidates = [
        ModelSpec(
            name='logistic_regression',
            pipeline=logreg_pipe,
            param_grid=logreg_param_grid,
            description='Logistic Regression with comprehensive regularisation'
        ),
        ModelSpec(
            name='svm', 
            pipeline=svm_pipe,
            param_grid=svm_param_grid,
            description='Support Vector Machine with optimised kernels'
        ),
        ModelSpec(
            name='random_forest',
            pipeline=rf_pipe,
            param_grid=rf_param_grid,
            description='Random Forest with ensemble optimisation'
        ),
        ModelSpec(
            name='gradient_boosting',
            pipeline=gb_pipe,
            param_grid=gb_param_grid,
            description='Gradient Boosting with learning optimisation'
        )
    ]
    
    return candidates


def get_unsupervised_candidates() -> List[ModelSpec]:
    """
    Get unsupervised model candidates for anomaly detection.
    Returns: List of ModelSpec objects for unsupervised learning
    """
    # Isolation Forest
    isolation_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", IsolationForest(random_state=42, n_jobs=-1)),
    ])

    # One Class SVM
    ocsvm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", OneClassSVM(kernel="rbf", gamma="scale")),
    ])

    candidates = [
        ModelSpec(
            name="isolation_forest",
            pipeline=isolation_pipe,
            param_grid={
                "classifier__n_estimators": [100, 200, 400],
                "classifier__contamination": [0.05, 0.1, 0.2],
            },
            description="Isolation Forest for anomaly detection",
        ),
        ModelSpec(
            name="one_class_svm",
            pipeline=ocsvm_pipe,
            param_grid={
                "classifier__nu": [0.01, 0.05, 0.1],
                "classifier__gamma": ["scale", 0.1, 0.01],
            },
            description="One-Class SVM (RBF) for anomaly detection",
        ),
    ]
    return candidates
