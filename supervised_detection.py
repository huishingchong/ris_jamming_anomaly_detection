#!/usr/bin/env python3
"""
Feature and training pipeline for RIS-based jamming detection (supervised learning)
Author: Hui Shing
"""

import argparse
import sys
import json
import logging
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_handler import DatasetPaths, load_features, extract_feature_matrix, prepare_binary_classification
from models import get_supervised_candidates
from timing import measure_inference_latency
from utils import set_random_seeds, setup_logging, make_json_serialisable
from metrics import ece, threshold_for_fpr, threshold_for_fnr, compute_comprehensive_metrics

warnings.filterwarnings("ignore")

DEFAULT_CONFIG = {
    # Policy targets
    "standard_target_fpr": 0.10,
    "standard_guard": 0.0,
    "stealthy_target_fnr": 0.10,
    "stealthy_guard": 0.0,

    # Feature selection
    "correlation_threshold": 0.95,

    # Sample weighting
    "stealthy_weight": 2.0,
    "enable_stealthy_weighting": True,

    # Calibration + tuning
    "calibration_method": "isotonic",
    "cv_folds": 5,
    "calib_size": 0.15,
    "val_size": 0.15,

    # Optional hyperparameter tuning
    "enable_tuning": False,
    "tuning_strategy": "grid", # grid or random
    "tuning_iter": 60, # for randomised search
    "tuning_scoring": "f1",

    "max_iter": 2000,
}

# Domain knowledge: protected and discriminative scores from each domain
DISCRIMINATIVE = ["std_magnitude", "mean_psd_db", "peak_to_avg_ratio", "received_power", "sinr_estimate"]

DISCRIMINATIVE_SCORES = {
    "std_magnitude": 0.684,
    "mean_magnitude": 0.683,
    "mean_psd_db": 0.621,
    "std_psd_db": 0.611,
    "received_power": 0.608,
    "sinr_estimate": 0.597,
    "spectral_flatness": 0.276,
    "spectral_entropy": 0.242,
    "peak_to_avg_ratio": 0.207,
    "amplitude_kurtosis": 0.199,
    "spectral_centroid": 0.184,
    "spectral_bandwidth": 0.112,
    "spectral_rolloff": 0.071,
    "mean_imag": 0.002,
    "mean_real": 0.000,
    "iq_power_ratio_db": 0.000,
}


def validate_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and validate the dataset"""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")
    if "label" not in df.columns:
        raise ValueError("Dataset must contain 'label' column")
    if "band_name" in df.columns:
        df["band_name"] = df["band_name"].fillna("none")
    return df


def compute_sample_weights(df: pd.DataFrame, stealthy_weight: float = 1.5, enable_weighting: bool = True) -> np.ndarray:
    """Compute sample weights with optional stealthy emphasis"""
    weights = np.ones(len(df))
    if enable_weighting and "band_name" in df.columns:
        stealthy_mask = (df["band_name"] == "stealthy") & (df["label"] == 1)
        weights[stealthy_mask] = stealthy_weight
        n_stealthy = int(stealthy_mask.sum())
        if n_stealthy > 0:
            logging.info(f"Applied {stealthy_weight}x weight to {n_stealthy} stealthy samples")
    return weights

def analyse_band_misclassifications(y_true: np.ndarray, y_pred: np.ndarray, metadata_df: pd.DataFrame, logger) -> Dict:
    """Analyse misclassifications by effectiveness band"""
    if "band_name" not in metadata_df.columns:
        return {}

    meta = metadata_df.copy()
    meta["misclassified"] = (y_true != y_pred)

    totals = meta.groupby("band_name")["misclassified"].size().rename("total_samples")
    miss = meta.groupby("band_name")["misclassified"].sum().rename("misclassified")

    report = pd.concat([totals, miss], axis=1).fillna(0).astype(int)
    report["error_rate"] = report["misclassified"] / report["total_samples"].clip(lower=1)

    logger.info("Band misclassification analysis:")
    for band, row in report.sort_index().iterrows():
        metric_name = "FPR" if band == "none" else "FNR"
        logger.info(f"  {band}: {row['misclassified']}/{row['total_samples']} ({row['error_rate']:.3f}) [{metric_name}]")

    return {
        band: {
            "misclassified": int(row["misclassified"]),
            "total_samples": int(row["total_samples"]),
            "error_rate": float(row["error_rate"]),
        }
        for band, row in report.iterrows()
    }


def measure_model_latency(model, X_sample: np.ndarray, n_runs: int = 30) -> float:
    try:
        latency_result = measure_inference_latency(model, X_sample, n_runs=n_runs, warmup_runs=10)
        return float(latency_result.median_ms)
    except Exception as e:
        logging.warning(f"Standard latency measurement failed: {e}, using fallback")

        sample_size = min(64, len(X_sample))
        X_timing = X_sample[:sample_size]

        for _ in range(5):  # warmup
            if hasattr(model, "predict_proba"):
                model.predict_proba(X_timing)
            else:
                model.predict(X_timing)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            if hasattr(model, "predict_proba"):
                model.predict_proba(X_timing)
            else:
                model.predict(X_timing)
            end = time.perf_counter()
            times.append((end - start) * 1000 / sample_size)

        return float(np.median(times))

def _split_two_way_safe(df: pd.DataFrame, frac: float, stratify_cols: List[str], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (train and heldout test) using stratification.
    Falls back to label-only if band-based stratification fails.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError(f"Split fraction must be in (0.0, 1.0), got {frac:.4f}")

    try:
        if len(stratify_cols) > 1 and "band_name" in stratify_cols:
            df_strat = df[stratify_cols].copy()
            for col in stratify_cols:
                df_strat[col] = df_strat[col].fillna("unknown")

            composite = df_strat[stratify_cols[0]].astype(str)
            for col in stratify_cols[1:]:
                composite = composite + "_" + df_strat[col].astype(str)

            train, heldout = train_test_split(
                df, test_size=frac, stratify=composite, random_state=seed
            )
            return train.reset_index(drop=True), heldout.reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Band stratification failed ({e}) - stratifying based on label only.")

    train, heldout = train_test_split(
        df, test_size=frac, stratify=df["label"], random_state=seed
    )
    return train.reset_index(drop=True), heldout.reset_index(drop=True)


class DetectionExperiment:
    """Feature selection and model evaluation pipeline for jamming detection"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_features(self, train_df: pd.DataFrame) -> List[str]:
        """
        Domain-informed correlation pruning:
        Start from numeric features (excluding label/band).
        Rank by: protected priority > discriminative score.
        Blocking highly correlated features (|r| >= threshold).
        """
        exclude = ["label", "band_name"]
        numeric = train_df.drop(columns=[c for c in exclude if c in train_df], errors="ignore")
        numeric = numeric.select_dtypes(include=[np.number])
        all_feats = list(numeric.columns)
        all_feats = [f for f in all_feats if f not in ['mean_imag', 'mean_real', 'iq_power_ratio_db']]

        if not all_feats:
            self.logger.warning("No numeric features found.")
            return []
        

        thr = float(self.config.get("correlation_threshold", 0.95))

        score_map = {**DISCRIMINATIVE_SCORES}
        def score_of(f: str) -> float:
            return float(score_map.get(f, 0.0))

        protected_set = set(f for f in DISCRIMINATIVE if f in all_feats)

        candidates = sorted(
            all_feats,
            key=lambda f: (f in protected_set, score_of(f)),
            reverse=True
        )

        # corr = numeric.corr().abs()
        corr = numeric[all_feats].corr().abs()
        kept: List[str] = []
        blocked: set = set()

        for f in candidates:
            if f in blocked:
                continue
            kept.append(f)
            # block highly-correlated companions
            if f in corr.columns:
                high_corr = corr.index[(corr[f] >= thr) & (corr.index != f)].tolist()
                blocked.update(high_corr)

        self.logger.info(f"Final features after domain-informed pruning ({len(kept)}): {kept}")
        return kept

    # Model Tune and calibration
    def _fit_with_optional_weights(self, model, X, y, sample_weights):
        """Fit model with sample_weight if supported"""
        try:
            if sample_weights is not None and np.any(sample_weights != 1.0):
                # estimator.fit signature
                fit_sig = getattr(model, "fit").__code__.co_varnames
                if "sample_weight" in fit_sig:
                    return model.fit(X, y, sample_weight=sample_weights)
                # pipeline last step
                if hasattr(model, "steps") and model.steps:
                    last_name = model.steps[-1][0]
                    return model.fit(X, y, **{f"{last_name}__sample_weight": sample_weights})
            return model.fit(X, y)
        except TypeError:
            self.logger.info("Estimator ignores sample_weight, so fitting without it.")
            return model.fit(X, y)

    def _maybe_tune(self, spec, X_train, y_train, sample_weights):
        """
        Optional hyperparameter tuning on the train split only. Returns a fitted "best" model.
        """
        enable = bool(self.config.get("enable_tuning", False))
        param_grid = getattr(spec, "param_grid", None)

        # No tuning path
        if (not enable) or (not param_grid):
            self.logger.info(f"  Tuning disabled or empty grid for {spec.name}, using default params.")
            model = clone(spec.pipeline)
            return self._fit_with_optional_weights(model, X_train, y_train, sample_weights)

        # CV setup
        cv_folds = int(self.config.get("cv_folds", 5))
        inner_cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = str(self.config.get("tuning_scoring", "f1"))

        # Choose searcher
        strategy = str(self.config.get("tuning_strategy", "grid")).lower()
        if strategy == "random":
            n_iter = int(self.config.get("tuning_iter", 60))
            search = RandomizedSearchCV(
                estimator=clone(spec.pipeline),
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
                random_state=42,
            )
        else:
            search = GridSearchCV(
                estimator=clone(spec.pipeline),
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )

        # Fit with sample_weight if supported
        fit_params = {}
        if hasattr(search.estimator, "steps") and search.estimator.steps:
            last_name = search.estimator.steps[-1][0]
            fit_params[f"{last_name}__sample_weight"] = sample_weights
        else:
            # direct estimator path
            fit_sig = getattr(search.estimator, "fit").__code__.co_varnames
            if "sample_weight" in fit_sig:
                fit_params["sample_weight"] = sample_weights

        try:
            search.fit(X_train, y_train, **fit_params)
            best_model = search.best_estimator_
            self.logger.info(f"  Best params for {spec.name}: {search.best_params_}")
            return best_model
        except Exception as e:
            self.logger.warning(f"  Tuning failed for {spec.name} ({e}), using default pipeline.")
            model = clone(spec.pipeline)
            return self._fit_with_optional_weights(model, X_train, y_train, sample_weights)

    def _calibrate_prefit(self, fitted_base_estimator, X_calib, y_calib) -> Tuple[CalibratedClassifierCV, str]:
        """
        Calibrate a prefit estimator on (X_calib, y_calib). Tries isotonic as default, otherwise sigmoid.
        Returns (calibrated_model, method_used).
        """
        method_pref = str(self.config.get("calibration_method", "isotonic"))
        methods = [method_pref] + ([m for m in ["isotonic", "sigmoid"] if m != method_pref])

        last_error = None
        for m in methods:
            try:
                cc = CalibratedClassifierCV(fitted_base_estimator, method=m, cv="prefit")
                cc.fit(X_calib, y_calib)
                return cc, m
            except Exception as e:
                last_error = e
                self.logger.warning(f"Calibration with '{m}' failed: {e}")

        raise RuntimeError(f"All calibration methods failed (last error: {last_error})")


    def train_and_evaluate_models(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, val_metadata: pd.DataFrame, sample_weights: np.ndarray, seed: int) -> Dict:
        """Train candidates, calibrate on calib split, select champion by policy and F1"""
        self.logger.info("CHAMPION SELECTION WITH POLICY-BASED THRESHOLDS")

        model_specs = get_supervised_candidates()
        results: Dict[str, Dict] = {}

        # Compliance cap aligned to how we pick threshold (target - guard)
        std_cap = max(0.0, float(self.config["standard_target_fpr"]) - float(self.config["standard_guard"]))

        for spec in model_specs:
            try:
                self.logger.info(f"Training {spec.name}...")

                # Train (with optional tuning) on the train split
                base_model = self._maybe_tune(spec, X_train, y_train, sample_weights)

                # Calibrate on the calibration split
                calibrated_model, calib_used = self._calibrate_prefit(base_model, X_calib, y_calib)

                # Validation probabilities for policy thresholding
                y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]

                # Policy thresholds for standard and stealthy
                thr_std, val_fpr = threshold_for_fpr(
                    y_val, y_val_proba,
                    target_fpr=self.config["standard_target_fpr"],
                    guard=self.config["standard_guard"]
                )

                thr_ste, val_fnr = threshold_for_fnr(
                    y_val, y_val_proba,
                    target_fnr=self.config["stealthy_target_fnr"],
                    guard=self.config["stealthy_guard"]
                )

                # Evaluate at standard threshold (for champion selection/overview)
                y_val_pred_std = (y_val_proba >= thr_std).astype(int)
                val_metrics_std = compute_comprehensive_metrics(y_val, y_val_pred_std, y_val_proba)
                val_fnr_std = val_metrics_std["fnr"]

                # Evaluate at stealthy threshold (to log/store its FPR & FNR)
                y_val_pred_ste = (y_val_proba >= thr_ste).astype(int)
                ste_val_metrics = compute_comprehensive_metrics(y_val, y_val_pred_ste, y_val_proba)
                
                val_fpr_std = float(val_metrics_std["fpr"])
                val_fnr_std = float(val_metrics_std["fnr"])
                val_fpr_ste = float(ste_val_metrics["fpr"])
                val_fnr_ste = float(ste_val_metrics["fnr"])

                latency_ms = measure_model_latency(calibrated_model, X_val)

                # Calibration audit
                val_ece = ece(y_val_proba, y_val)
                val_brier = brier_score_loss(y_val, y_val_proba)

                results[spec.name] = {
                    "calibrated_model": calibrated_model,
                    "thresholds": {
                        "standard": {"threshold": float(thr_std), "val_fpr": float(val_fpr_std), "val_fnr": float(val_fnr_std)},
                        "stealthy": {"threshold": float(thr_ste), "val_fnr": float(val_fnr_ste), "val_fpr": float(val_fpr_ste)},
                    },
                    # "stealthy_info": stealthy_info,
                    "val_metrics": val_metrics_std,  # keep champion selection on the standard policy view
                    "calibration_audit": {
                        "method": calib_used,
                        "val_ece": float(val_ece),
                        "val_brier": float(val_brier),
                    },
                    "latency_ms": float(latency_ms),
                }

                self.logger.info(
                    f"  {spec.name}: "
                    f"std_thr={thr_std:.3f} (FPR={val_fpr_std:.3f}, FNR={val_fnr_std:.3f}) | "
                    f"ste_thr={thr_ste:.3f} (FPR={val_fpr_ste:.3f}, FNR={val_fnr_ste:.3f}) | "
                    f"val_f1={val_metrics_std['f1']:.3f}, latency={latency_ms:.2f}ms"
                )

            except Exception as e:
                self.logger.warning(f"  {spec.name} failed: {e}", exc_info=False)
                continue

        if not results:
            raise RuntimeError("All models failed during training")

        # Eligibility by FPR compliance using the same cap used for threshold selection
        eligible = [(n, r) for (n, r) in results.items() if np.isfinite(r["thresholds"]["standard"]["val_fpr"])
                    and (r["thresholds"]["standard"]["val_fpr"] <= std_cap)]

        if not eligible:
            self.logger.warning("No models meet FPR target cap - selecting best available by F1.")
            eligible = list(results.items())

        # Champion: highest F1, tie-breaker lowest latency
        champion_name, champion_result = max(
            eligible, key=lambda x: (x[1]["val_metrics"]["f1"], -x[1]["latency_ms"])
        )
        self.logger.info(f"Selected champion: {champion_name}")

        return {
            "champion_name": champion_name,
            "champion_result": champion_result,
            "all_results": {name: {k: v for k, v in res.items() if k != "calibrated_model"}
                            for name, res in results.items()},
            "n_candidates": len(model_specs),
        }
    

    def evaluate_test_performance(self, champion_result: Dict, X_test: np.ndarray, y_test: np.ndarray, test_metadata: pd.DataFrame) -> Dict:
        """Final evaluation on test set with both thresholds"""
        calibrated_model = champion_result["calibrated_model"]
        thr_std = float(champion_result["thresholds"]["standard"]["threshold"])
        thr_ste = float(champion_result["thresholds"]["stealthy"]["threshold"])

        y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]

        # Standard policy
        y_std = (y_test_proba >= thr_std).astype(int)
        std_metrics = compute_comprehensive_metrics(y_test, y_std, y_test_proba)
        std_bands = analyse_band_misclassifications(y_test, y_std, test_metadata, self.logger)

        # Stealthy policy
        y_ste = (y_test_proba >= thr_ste).astype(int)
        ste_metrics = compute_comprehensive_metrics(y_test, y_ste, y_test_proba)
        ste_bands = analyse_band_misclassifications(y_test, y_ste, test_metadata, self.logger)

        # Calibration audit on test
        test_ece = ece(y_test_proba, y_test)
        test_brier = brier_score_loss(y_test, y_test_proba)

        # Drift relative to validation operating points
        val_fpr = float(champion_result["thresholds"]["standard"]["val_fpr"])
        val_fnr = float(champion_result["thresholds"]["stealthy"]["val_fnr"])
        drift = {
            "standard": {"val_to_test_fpr_drift": float(std_metrics["fpr"] - val_fpr) if np.isfinite(std_metrics["fpr"]) else float("nan")},
            "stealthy": {"val_to_test_fnr_drift": float(ste_metrics["fnr"] - val_fnr) if np.isfinite(ste_metrics["fnr"]) else float("nan")},
        }

        self.logger.info("Standard Policy Results:")
        self.logger.info(f"  Threshold: {thr_std:.3f}")
        self.logger.info(
            f"  Test FPR: {std_metrics['fpr']:.4f} | Test FNR: {std_metrics['fnr']:.4f} "
            f"(FPR drift from val: {std_metrics['fpr'] - champion_result['thresholds']['standard']['val_fpr']:+.4f})"
        )
        self.logger.info(f"  Test F1: {std_metrics['f1']:.4f}")

        self.logger.info("Stealthy Policy Results:")
        self.logger.info(f"  Threshold: {thr_ste:.3f}")
        self.logger.info(
            f"  Test FPR: {ste_metrics['fpr']:.4f} | Test FNR: {ste_metrics['fnr']:.4f} "
            f"(FNR drift from val: {ste_metrics['fnr'] - champion_result['thresholds']['stealthy']['val_fnr']:+.4f})"
        )
        self.logger.info(f"  Test F1: {ste_metrics['f1']:.4f}")

        self.logger.info(f"Calibration: ECE={test_ece:.4f}, Brier={test_brier:.4f}")

        # Optional bootstrap confidence intervals (both policies)
        boot_n = int(self.config.get("bootstrap_iters", 0))
        ci_std = None
        ci_ste = None
        if boot_n > 0:
            rng = np.random.default_rng(42)
            n = len(y_test)

            def _ci_for_threshold(thr):
                f1s, fprs, fnrs, recs, precs, accs = [], [], [], [], [], []
                for _ in range(boot_n):
                    idx = rng.integers(0, n, size=n)
                    yt = y_test[idx]
                    ys = y_test_proba[idx]
                    yhat = (ys >= thr).astype(int)
                    m = compute_comprehensive_metrics(yt, yhat, ys)
                    f1s.append(m["f1"]); fprs.append(m["fpr"]); fnrs.append(m["fnr"]); recs.append(m["recall"]); precs.append(m["precision"]); accs.append(m["accuracy"])
                def stats(arr):
                    a = np.array(arr, dtype=float)
                    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "lower_95": float(np.nanpercentile(a, 2.5)), "upper_95": float(np.nanpercentile(a, 97.5))}
                return {
                    "f1": stats(f1s),
                    "fpr": stats(fprs),
                    "fnr": stats(fnrs),
                    "recall": stats(recs),
                    "precision": stats(precs),
                    "accuracy": stats(accs),
                }

            ci_std = _ci_for_threshold(thr_std)
            ci_ste = _ci_for_threshold(thr_ste)

        return {
            "standard": {
                "test_metrics": std_metrics,
                "test_band_analysis": std_bands,
                "threshold": thr_std,
                "policy_target": self.config["standard_target_fpr"],
                "bootstrap_ci": ci_std,
            },
            "stealthy": {
                "test_metrics": ste_metrics,
                "test_band_analysis": ste_bands,
                "threshold": thr_ste,
                "policy_target": self.config["stealthy_target_fnr"],
                "bootstrap_ci": ci_ste,
            },
            "calibration_audit": {"test_ece": test_ece, "test_brier": test_brier},
            "drift_analysis": drift,
        }

    def get_feature_importance(self, model: CalibratedClassifierCV, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from calibrated model if available"""
        try:
            cc = model.calibrated_classifiers_[0]
            base = getattr(cc, "base_estimator", getattr(cc, "estimator", None))
            if base is None:
                return {}

            # If pipeline, try the 'classifier' step
            clf = base
            if hasattr(base, "named_steps") and "classifier" in base.named_steps:
                clf = base.named_steps["classifier"]

            if hasattr(clf, "coef_") and clf.coef_ is not None:
                coefs = np.ravel(clf.coef_)
                imp = dict(zip(feature_names, np.abs(coefs)))
                return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

            if hasattr(clf, "feature_importances_"):
                imp = dict(zip(feature_names, clf.feature_importances_))
                return dict(sorted(imp.items(), key=lambda x: x[1], reverse=True))

            return {}
        except Exception as e:
            logging.warning(f"Could not extract feature importance: {e}")
            return {}

    def save_experiment_results(self, results: Dict, output_dir: Path, seed: int):
        """save model artifacts and JSON summaries"""
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Keep legacy flat fields *and* structured config to avoid breaking downstream consumers.
        model_artifacts = {
            'model': results["model"],
            'model_type': 'supervised',
            'final_features': results["final_features"],
            'champion_name': results["champion_name"],
            'thresholds': {
                'standard': results["test_results"]["standard"]["threshold"],
                'stealthy': results["test_results"]["stealthy"]["threshold"]
            },
            'validation_metrics': results.get("champion_result", {}).get("val_metrics", {}),
            'latency_ms': float(results["val_latency_ms"]),
            'calibration_audit': results.get("champion_result", {}).get("calibration_audit"),
            'policy_config': {
                "standard_target_fpr": self.config["standard_target_fpr"],
                "stealthy_target_fnr": self.config["stealthy_target_fnr"],
                "calibration_method": self.config["calibration_method"],
            },
            'config': self.config,
        }
        joblib.dump(model_artifacts, seed_dir / "model_artifacts.joblib")

        with open(seed_dir / "selected_features.txt", "w") as f:
            f.write("\n".join(results["final_features"]))

        json_results = make_json_serialisable({k: v for k, v in results.items() if k != "model"})
        with open(seed_dir / "experiment_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        self.logger.info(f"Results saved to: {seed_dir}")


    def run_single_experiment(self, df: pd.DataFrame, seed: int) -> Dict:
        """End-to-end run for a single seed"""
        self.logger.info(f"Starting detection experiment with seed {seed}")
        set_random_seeds(seed)
        attack_class = 1 # 1 = Binary detection of RIS jamming, 2 = Binary detection of Conventional jamming
        binary_df = prepare_binary_classification(df, normal_class=0, attack_class=attack_class)

        # Four-way split fractions
        test_abs = 0.20
        remain = 1.0 - test_abs
        calib_abs = float(self.config["calib_size"])
        val_abs = float(self.config["val_size"])

        if not (0.0 < calib_abs < remain):
            raise ValueError(f"Invalid calib_size {calib_abs} - must be in (0, {remain}).")
        remain_after_calib = remain - calib_abs
        if remain_after_calib <= 0:
            raise ValueError("Invalid split configuration: remaining mass after calib is non-positive.")
        if not (0.0 < val_abs < remain_after_calib):
            raise ValueError(f"Invalid val_size {val_abs} - must be in (0, {remain_after_calib}).")

        # Convert desired absolute masses to relative fractions for two-way helper
        calib_adj = calib_abs / remain
        val_adj = val_abs / remain_after_calib
        if not (0.0 < calib_adj < 1.0) or not (0.0 < val_adj < 1.0):
            raise ValueError(f"Derived split sizes invalid: calib={calib_adj:.4f}, val={val_adj:.4f}")

        strat_cols = ["label", "band_name"] if "band_name" in binary_df.columns else ["label"]

        # (train_full, test)
        train_full_df, test_df = _split_two_way_safe(binary_df, test_abs, strat_cols, seed)
        # (train_after_calib, calib)
        train_calib_df, calib_df = _split_two_way_safe(train_full_df, calib_adj, strat_cols, seed)
        # (train, val)
        train_df, val_df = _split_two_way_safe(train_calib_df, val_adj, strat_cols, seed)

        # Prepare features
        final_features = self.prepare_features(train_df)
        
        # This is for active jamming exploration only
        exploration = False
        if attack_class == 2 and exploration:
            # We are running active jamming exploring, compare same feature as what was identified in RIS jamming to compare
            final_features = ['std_magnitude', 'received_power', 'peak_to_avg_ratio', 'spectral_flatness', 'amplitude_kurtosis', 'spectral_bandwidth', 'spectral_rolloff']
        
        X_train, y_train, train_names = extract_feature_matrix(train_df)
        X_calib, y_calib, calib_names = extract_feature_matrix(calib_df)
        X_val, y_val, val_names = extract_feature_matrix(val_df)
        X_test, y_test, test_names = extract_feature_matrix(test_df)

        if not (train_names == calib_names == val_names == test_names):
            self.logger.warning("Feature names inconsistent across splits, using training feature list for selection.")

        # Use the chosen feature list (intersection with available)
        available = [f for f in final_features if f in train_names]
        if len(available) != len(final_features):
            missing = set(final_features) - set(available)
            self.logger.warning(f"Some selected features not found in data: {missing}")
            final_features = available

        # Slice matrices to selected features
        feat_idx = [train_names.index(f) for f in final_features]
        X_train_sel = X_train[:, feat_idx]
        X_calib_sel = X_calib[:, feat_idx]
        X_val_sel = X_val[:, feat_idx]
        X_test_sel = X_test[:, feat_idx]

        # Stealthy weighting on training split only
        train_weights = compute_sample_weights(
            train_df, self.config["stealthy_weight"], self.config["enable_stealthy_weighting"]
        )

        self.logger.info(
            f"Data splits: train={len(X_train_sel)}, calib={len(X_calib_sel)}, "
            f"val={len(X_val_sel)}, test={len(X_test_sel)}"
        )
        if "band_name" in binary_df.columns:
            for s_name, s_df in [("Train", train_df), ("Calib", calib_df), ("Val", val_df), ("Test", test_df)]:
                band_dist = s_df["band_name"].value_counts().to_dict()
                self.logger.info(f"{s_name} band distribution: {band_dist}")

        # Train, select champion
        champion = self.train_and_evaluate_models(
            X_train_sel, y_train, X_calib_sel, y_calib, X_val_sel, y_val, val_df, train_weights, seed
        )

        # Test evaluation
        test_results = self.evaluate_test_performance(
            champion["champion_result"], X_test_sel, y_test, test_df
        )

        feat_importance = self.get_feature_importance(
            champion["champion_result"]["calibrated_model"], final_features
        )

        return {
            "seed": seed,
            "final_features": final_features,
            "n_features": len(final_features),
            "champion_name": champion["champion_name"],
            "model": champion["champion_result"]["calibrated_model"],
            "test_results": test_results,
            "feature_importance": feat_importance,
            "model_comparison": champion["all_results"],
            "data_splits": {
                "train_size": len(X_train_sel),
                "calib_size": len(X_calib_sel),
                "val_size": len(X_val_sel),
                "test_size": len(X_test_sel),
            },
            "val_latency_ms": float(champion["champion_result"]["latency_ms"]),
        }


def compute_aggregate_statistics(all_results: List[Dict], config: Dict) -> Dict:
    """Aggregate statistics across seeds for both normal threshold and stealthy optimised threshold"""
    std_metrics = [r["test_results"]["standard"]["test_metrics"] for r in all_results]
    ste_metrics = [r["test_results"]["stealthy"]["test_metrics"] for r in all_results]

    def agg(metrics_list, prefix):
        f1s = [m["f1"] for m in metrics_list]
        accs = [m["accuracy"] for m in metrics_list]
        fprs = [m["fpr"] for m in metrics_list]
        recs = [m["recall"] for m in metrics_list]
        fnrs = [m["fnr"] for m in metrics_list]
        return {
            f"{prefix}_f1_score": {"mean": float(np.nanmean(f1s)), "std": float(np.nanstd(f1s))},
            f"{prefix}_accuracy": {"mean": float(np.nanmean(accs)), "std": float(np.nanstd(accs))},
            f"{prefix}_fpr": {"mean": float(np.nanmean(fprs)), "std": float(np.nanstd(fprs))},
            f"{prefix}_recall": {"mean": float(np.nanmean(recs)), "std": float(np.nanstd(recs))},
            f"{prefix}_fnr": {"mean": float(np.nanmean(fnrs)), "std": float(np.nanstd(fnrs))},
        }

    std_fpr_compliant = sum(
        1 for r in all_results
        if np.isfinite(r["test_results"]["standard"]["test_metrics"]["fpr"])
        and r["test_results"]["standard"]["test_metrics"]["fpr"] <= config["standard_target_fpr"]
    )
    ste_fnr_compliant = sum(
        1 for r in all_results
        if np.isfinite(r["test_results"]["stealthy"]["test_metrics"]["fnr"])
        and r["test_results"]["stealthy"]["test_metrics"]["fnr"] <= config["stealthy_target_fnr"]
    )

    calib = [r["test_results"]["calibration_audit"] for r in all_results]
    eces = [c["test_ece"] for c in calib]
    briers = [c["test_brier"] for c in calib]

    feat_counts = [r["n_features"] for r in all_results]
    latencies = [r.get("val_latency_ms", float("nan")) for r in all_results]

    return make_json_serialisable({
        "experiment_summary": {"n_seeds": len(all_results), "successful_runs": len(all_results)},
        "standard_threshold_performance": agg(std_metrics, "standard"),
        "stealthy_threshold_performance": agg(ste_metrics, "stealthy"),
        "policy_compliance": {
            "standard_fpr_compliant": int(std_fpr_compliant),
            "stealthy_fnr_compliant": int(ste_fnr_compliant),
            "total_seeds": len(all_results),
        },
        "calibration_quality": {
            "ece": {"mean": float(np.nanmean(eces)), "std": float(np.nanstd(eces))},
            "brier": {"mean": float(np.nanmean(briers)), "std": float(np.nanstd(briers))},
        },
        "operational_metrics": {
            "feature_count": {"mean": float(np.nanmean(feat_counts)), "std": float(np.nanstd(feat_counts))},
            "val_latency_ms": {"mean": float(np.nanmean(latencies)), "std": float(np.nanstd(latencies))},
        },
    })


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection and model training with policy-based thresholds"
    )
    parser.add_argument("--csv", required=True, help="Path to dataset CSV file")
    parser.add_argument("--output", default="experimental_results/supervised", help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 999], help="Random seeds for multiple runs")

    # Policy
    parser.add_argument("--standard-target-fpr", type=float, default=0.10, help="Target FPR for standard policy")
    parser.add_argument("--stealthy-target-fnr", type=float, default=0.10, help="Target FNR for stealthy policy")
    parser.add_argument("--standard-guard", type=float, default=0.0, help="Guard band for standard FPR")
    parser.add_argument("--stealthy-guard", type=float, default=0.0, help="Guard band for stealthy FNR")
    parser.add_argument("--stealthy-heuristic", action="store_true",
                        help="Use stealthy-aware heuristic threshold (guarded by standard FPR cap) instead of FNR policy")
    parser.add_argument("--bootstrap-iters", type=int, default=0,
                        help="If >0, compute bootstrap CIs on test metrics with this many resamples")

    # Weighting + calibration splits
    parser.add_argument("--stealthy-weight", type=float, default=2.0, help="Weight multiplier for stealthy positives")
    parser.add_argument("--disable-stealthy-weighting", action="store_true", help="Disable stealthy weighting")
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"], default="isotonic",
                        help="Preferred calibration method")
    parser.add_argument("--calib-size", type=float, default=0.15,
                        help="Calibration set size (absolute fraction of original, before val split)")
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Validation set size (absolute fraction of original, after calib)")

    # Model tuning
    parser.add_argument("--enable-tuning", action="store_true", help="Enable hyperparameter tuning on the train split")
    parser.add_argument("--tuning-strategy", choices=["grid", "random"], default="grid", help="Tuning strategy")
    parser.add_argument("--tuning-iter", type=int, default=60, help="n_iter for randomized search")
    parser.add_argument("--tuning-scoring", default="f1", help="sklearn scoring string (default=f1)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", output_dir / "supervised_detection.log")
    logger = logging.getLogger(__name__)

    config = DEFAULT_CONFIG.copy()
    config.update({
        "standard_target_fpr": args.standard_target_fpr,
        "stealthy_target_fnr": args.stealthy_target_fnr,
        "standard_guard": args.standard_guard,
        "stealthy_guard": args.stealthy_guard,
        "stealthy_weight": args.stealthy_weight,
        "enable_stealthy_weighting": not args.disable_stealthy_weighting,
        "calibration_method": args.calibration_method,
        "calib_size": args.calib_size,
        "val_size": args.val_size,
        "enable_tuning": bool(args.enable_tuning),
        "tuning_strategy": args.tuning_strategy,
        "tuning_iter": args.tuning_iter,
        "tuning_scoring": args.tuning_scoring,
        "output_dir": str(output_dir),
        "stealthy_heuristic": bool(args.stealthy_heuristic),
        "bootstrap_iters": int(args.bootstrap_iters),
    })

    logger.info(f"Detection Configuration: {config}")
    logger.info(f"Seeds: {args.seeds}")

    csv_path = Path(args.csv)
    try:
        df = validate_dataset(csv_path)
        logger.info(f"Loaded dataset: {len(df)} samples, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return 1

    experiment = DetectionExperiment(config)
    all_results: List[Dict] = []

    for seed in args.seeds:
        try:
            result = experiment.run_single_experiment(df, seed)
            all_results.append(result)
            experiment.save_experiment_results(result, output_dir, seed)
        except Exception as e:
            logger.error(f"Experiment failed for seed {seed}: {e}", exc_info=True)

    if not all_results:
        logger.error("All experiments failed")
        return 1

    stats = compute_aggregate_statistics(all_results, config)
    
    summary = {
        "experiment_name": "Detection_Policy_Pipeline",
        "dataset_path": str(csv_path),
        "config": config,
        "aggregate_statistics": stats,
        "per_seed_bootstrap": [
            {
                "seed": r.get("seed"),
                "standard": r.get("test_results", {}).get("standard", {}).get("bootstrap_ci"),
                "stealthy": r.get("test_results", {}).get("stealthy", {}).get("bootstrap_ci"),
            } for r in all_results
        ]
    }
    with open(output_dir / "detection_summary.json", "w") as f:
        json.dump(make_json_serialisable(summary), f, indent=2)

    std_perf = stats["standard_threshold_performance"]
    ste_perf = stats["stealthy_threshold_performance"]
    policy_comp = stats["policy_compliance"]
    cal_qual = stats["calibration_quality"]
    ops = stats["operational_metrics"]
    
    print("\nPolicy-Based Pipeline Results Summary")
    print(f"Dataset: {csv_path.name}")
    print(f"Successful runs: {stats['experiment_summary']['successful_runs']}/{len(args.seeds)}")

    print(f"\nStandard Policy Performance (FPR ≤ {config['standard_target_fpr']:.2f}):")
    print(f"  F1 Score: {std_perf['standard_f1_score']['mean']:.3f} ± {std_perf['standard_f1_score']['std']:.3f}")
    print(f"  Accuracy: {std_perf['standard_accuracy']['mean']:.3f} ± {std_perf['standard_accuracy']['std']:.3f}")
    print(f"  FPR: {std_perf['standard_fpr']['mean']:.3f} ± {std_perf['standard_fpr']['std']:.3f}")
    print(f"  FNR: {std_perf['standard_fnr']['mean']:.3f} ± {std_perf['standard_fnr']['std']:.3f}")
    print(f"  Policy compliance: {policy_comp['standard_fpr_compliant']}/{policy_comp['total_seeds']} seeds")

    print(f"\nStealthy Policy Performance (FNR ≤ {config['stealthy_target_fnr']:.2f}):")
    print(f"  F1 Score: {ste_perf['stealthy_f1_score']['mean']:.3f} ± {ste_perf['stealthy_f1_score']['std']:.3f}")
    print(f"  Accuracy: {ste_perf['stealthy_accuracy']['mean']:.3f} ± {ste_perf['stealthy_accuracy']['std']:.3f}")
    print(f"  FPR: {ste_perf['stealthy_fpr']['mean']:.3f} ± {ste_perf['stealthy_fpr']['std']:.3f}")
    print(f"  FNR: {ste_perf['stealthy_fnr']['mean']:.3f} ± {ste_perf['stealthy_fnr']['std']:.3f}")
    print(f"  Policy compliance: {policy_comp['stealthy_fnr_compliant']}/{policy_comp['total_seeds']} seeds")

    print("\nCalibration Quality:")
    print(f"  ECE: {cal_qual['ece']['mean']:.4f} ± {cal_qual['ece']['std']:.4f}")
    print(f"  Brier Score: {cal_qual['brier']['mean']:.4f} ± {cal_qual['brier']['std']:.4f}")

    print("\nOperational Characteristics:")
    print(f"  Feature count: {ops['feature_count']['mean']:.1f} ± {ops['feature_count']['std']:.1f}")
    print(f"  val_latency_ms: {ops['val_latency_ms']['mean']:.4f} ± {ops['val_latency_ms']['std']:.4f}")

    print(f"\nResults saved to: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
