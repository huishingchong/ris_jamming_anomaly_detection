"""
Purpose: Aggregate per-seed evaluation outputs for RIS-jamming experiments across standard and stealthy threshold modes.
Inputs: --standard-root and --stealthy-root containing seed_*/results/evaluation_complete_results.json
Outputs: Comprehensive CSVs and charts for dissertation results chapter - output lots of views and can analyse later
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
BAND_ORDER = ["ultra-stealthy", "stealthy", "moderate", "severe", "critical", "ultra-strong"]

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def band_from_name(name: str) -> str:
    """Map dataset name to band."""
    if not name:
        return "unknown"
    s = str(name).lower().replace("_", "-")
    tokens = s.split("-")

    # Word-based detection
    if "ultra" in tokens and "stealthy" in tokens:
        return "ultra-stealthy"
    if "ultra" in tokens and "strong" in tokens:
        return "ultra-strong"
    if "critical" in tokens:
        return "critical"
    if "severe" in tokens:
        return "severe"
    if "moderate" in tokens:
        return "moderate"
    if "stealthy" in tokens:
        return "stealthy"

    # Numeric pattern matching
    patterns = [
        (["22-25", "22_25"], "ultra-strong"),
        (["15-22", "15_22"], "critical"),
        (["10-15", "10_15"], "severe"),
        (["6-10", "6_10"], "moderate"),
        (["3-6", "3_6"], "stealthy"),
        (["1-3", "1_3"], "ultra-stealthy"),
    ]
    for pattern_list, band in patterns:
        for pat in pattern_list:
            if pat in s:
                return band
    return "unknown"

def scan_mode_dir(root: Path, mode: str) -> pd.DataFrame:
    """Parse per-seed evaluation_complete_results.json and produce rows of metrics."""
    
    # Metrics to have
    cols = [
        "seed", "mode", "dataset_name", "band",
        "accuracy", "precision", "recall", "f1",
        "fpr", "fnr", "tpr", "tnr",
        "roc_auc", "pr_auc",
        "macro_f1", "balanced_accuracy", "weighted_f1"
    ]
    rows: List[Dict[str, Any]] = []

    if not root or not root.exists():
        return pd.DataFrame(columns=cols)

    for seed_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        # Support both path structures
        candidates = [
            seed_dir / "results" / "evaluation_complete_results.json",
            seed_dir / "evaluation_complete_results.json",
        ]
        file = None
        for c in candidates:
            if c.exists():
                file = read_json(c)
                break
        if not file:
            continue

        evals = file.get("evaluation_results", [])
        for e in evals:
            dsname = e.get("dataset_name", "")
            band = band_from_name(dsname)

            perf = e.get("performance", {}) or {}
            metrics = perf.get("comprehensive_metrics", {}) or {}

            rows.append({
                "seed": int(seed_dir.name.replace("seed_", "")),
                "mode": mode,
                "dataset_name": dsname,
                "band": band,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "fpr": metrics.get("fpr"),
                "fnr": metrics.get("fnr"),
                "tpr": metrics.get("tpr"),
                "tnr": metrics.get("tnr"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "macro_f1": metrics.get("macro_f1"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "weighted_f1": metrics.get("weighted_f1"),
            })

    df = pd.DataFrame(rows, columns=cols)
    if df.empty:
        return df

    df["band"] = df["band"].apply(band_from_name)
    df = df[df["band"].isin(BAND_ORDER)].copy()
    return df

def scan_bootstrap_auc(root: Path, mode: str) -> pd.DataFrame:
    """
    Collect bootstrap AUC CI summaries from evaluation_complete_results.json for each seed/dataset.
    Returns columns: seed, mode, dataset_name, roc_auc_mean, roc_auc_lower95, roc_auc_upper95,
    pr_auc_mean, pr_auc_lower95, pr_auc_upper95, iters
    """
    cols = ["seed", "mode", "dataset_name",
            "roc_auc_mean", "roc_auc_lower95", "roc_auc_upper95",
            "pr_auc_mean", "pr_auc_lower95", "pr_auc_upper95", "iters"]
    rows = []

    if not root or not root.exists():
        return pd.DataFrame(columns=cols)

    for seed_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        candidates = [
            seed_dir / "results" / "evaluation_complete_results.json",
            seed_dir / "evaluation_complete_results.json",
        ]
        file = next((c for c in candidates if c.exists()), None)
        if file is None:
            continue

        j = read_json(file)
        if not j:
            continue

        for ds in j.get("evaluation_results", []):
            ds_name = ds.get("dataset_name")
            ba = (ds.get("performance", {}) or {}).get("bootstrap_auc", None)
            if not ba:
                continue
            roc = ba.get("roc_auc") or {}
            pr  = ba.get("pr_auc") or {}
            rows.append({
                "seed": int(seed_dir.name.replace("seed_", "")),
                "mode": mode,
                "dataset_name": ds_name,
                "roc_auc_mean": roc.get("mean"),
                "roc_auc_lower95": roc.get("lower_95"),
                "roc_auc_upper95": roc.get("upper_95"),
                "pr_auc_mean": pr.get("mean"),
                "pr_auc_lower95": pr.get("lower_95"),
                "pr_auc_upper95": pr.get("upper_95"),
                "iters": ba.get("iters"),
            })

    return pd.DataFrame(rows, columns=cols)

def scan_threshold_ci(root: Path, mode: str) -> pd.DataFrame:
    """
    Collect bootstrapped CIs for operating-point metrics (f1, fpr, fnr, recall, precision, accuracy)
    from evaluation_complete_results.json.
    Returns long-form rows: seed, mode, dataset_name, metric, mean, std, lower_95, upper_95.
    """
    cols = ["seed", "mode", "dataset_name", "metric", "mean", "std", "lower_95", "upper_95"]
    rows = []
    if not root or not root.exists():
        return pd.DataFrame(columns=cols)

    for seed_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        candidates = [
            seed_dir / "results" / "evaluation_complete_results.json",
            seed_dir / "evaluation_complete_results.json",
        ]
        file = next((c for c in candidates if c.exists()), None)
        if file is None:
            continue

        j = read_json(file)
        if not j:
            continue

        seed_val = int(seed_dir.name.replace("seed_", ""))
        for ds in j.get("evaluation_results", []):
            ds_name = ds.get("dataset_name", "")
            ci = (ds.get("performance", {}) or {}).get("confidence_intervals", None)
            if not ci:
                continue
            for metric_name, stats in ci.items():
                rows.append({
                    "seed": seed_val,
                    "mode": mode,
                    "dataset_name": ds_name,
                    "metric": metric_name,
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                    "lower_95": stats.get("lower_95"),
                    "upper_95": stats.get("upper_95"),
                })

    return pd.DataFrame(rows, columns=cols)

def scan_latency(root: Path, mode: str) -> pd.DataFrame:
    """
    Collect test latency summaries (median ms/sample) from evaluation_complete_results.json.
    Returns columns: seed, mode, dataset_name, median_ms_per_sample
    """
    cols = ["seed", "mode", "dataset_name", "median_ms_per_sample"]
    rows = []

    if not root or not root.exists():
        return pd.DataFrame(columns=cols)

    for seed_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        candidates = [
            seed_dir / "results" / "evaluation_complete_results.json",
            seed_dir / "evaluation_complete_results.json",
        ]
        file = next((c for c in candidates if c.exists()), None)
        if file is None:
            continue

        j = read_json(file)
        if not j:
            continue

        for ds in j.get("evaluation_results", []):
            ds_name = ds.get("dataset_name")
            lat = (ds.get("performance", {}) or {}).get("test_latency", None)
            if not lat:
                continue
            rows.append({
                "seed": int(seed_dir.name.replace("seed_", "")),
                "mode": mode,
                "dataset_name": ds_name,
                "median_ms_per_sample": lat.get("median_ms_per_sample"),
            })

    return pd.DataFrame(rows, columns=cols)

def read_champion_and_val_latency(seed_dir: Path) -> Tuple[Optional[str], Optional[float]]:
    """Extract champion model name and validation latency from experiment_results.json."""
    j = read_json(seed_dir / "experiment_results.json")
    if not j:
        return None, None

    champion = j.get("champion_name")
    # Look for val_latency_ms in multiple locations
    lat = None
    if "val_latency_ms" in j:
        lat = j["val_latency_ms"]
    elif "champion_result" in j:
        lat = j["champion_result"].get("latency_ms")
    
    try:
        return champion, (float(lat) if lat is not None else None)
    except Exception:
        return champion, None

def build_seed_meta(standard_root: Path) -> pd.DataFrame:
    """Build per-seed metadata from STANDARD runs."""
    rows: List[Dict[str, Any]] = []
    if not standard_root or not standard_root.exists():
        return pd.DataFrame(columns=["seed", "champion", "val_latency_ms"])

    for seed_dir in sorted([p for p in standard_root.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
        champion, lat = read_champion_and_val_latency(seed_dir)
        rows.append({
            "seed": int(seed_dir.name.replace("seed_", "")),
            "champion": champion,
            "val_latency_ms": lat,
        })
    return pd.DataFrame(rows, columns=["seed", "champion", "val_latency_ms"])

def macro_by_seed(df_long: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight macro across bands per (mode, seed)."""
    agg_dict = {
        "accuracy": ("accuracy", "mean"),
        "precision": ("precision", "mean"),
        "recall": ("recall", "mean"),
        "f1": ("f1", "mean"),
        "fpr": ("fpr", "mean"),
        "fnr": ("fnr", "mean"),
        "tpr": ("tpr", "mean"),
        "tnr": ("tnr", "mean"),
        "roc_auc": ("roc_auc", "mean"),
        "pr_auc": ("pr_auc", "mean"),
        "macro_f1": ("macro_f1", "mean"),
    }
    
    return (
        df_long.groupby(["mode", "seed"])
               .agg(**agg_dict)
               .reset_index()
    )

def macro_overall(macro_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds."""
    agg_dict = {}
    metrics = ["accuracy","precision","recall","f1","fpr","fnr","tpr","tnr","roc_auc","pr_auc","macro_f1"]
    
    for metric in metrics:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")
    
    return (
        macro_seed.groupby("mode")
                  .agg(**agg_dict)
                  .reset_index()
    )

def per_band(df_long: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Per-band means/std across seeds for one mode."""
    d = df_long[df_long["mode"] == mode]
    
    agg_dict = {}
    metrics = ["accuracy","precision","recall","f1","fpr","fnr","roc_auc","pr_auc","macro_f1"]
    
    for metric in metrics:
        agg_dict[f"{metric}_mean"] = (metric, "mean")
        agg_dict[f"{metric}_std"] = (metric, "std")
    
    out = (
        d.groupby("band")
         .agg(**agg_dict)
         .reindex(BAND_ORDER)
         .reset_index()
    )
    return out

def delta_by_band(std_band: pd.DataFrame, ste_band: pd.DataFrame) -> pd.DataFrame:
    """(stealthy - standard) deltas for band means."""
    mcols = ["accuracy_mean", "f1_mean", "recall_mean", "precision_mean", "fpr_mean"]
    std = std_band.set_index("band")[mcols]
    ste = ste_band.set_index("band")[mcols]
    delt = (ste - std).reindex(BAND_ORDER)
    delt.columns = [c.replace("_mean", "") + "_delta" for c in delt.columns]
    return delt.reset_index()

def integrity_report(df_long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (mode, seed), grp in df_long.groupby(["mode", "seed"]):
        bands = grp["band"].tolist()
        counts = grp["band"].value_counts().to_dict()
        missing = [b for b in BAND_ORDER if b not in counts]
        dupes = {b: c for b, c in counts.items() if c > 1}
        rows.append({
            "mode": mode,
            "seed": seed,
            "n_rows": int(len(grp)),
            "bands_found": ";".join([b for b in BAND_ORDER if b in counts]),
            "missing_bands": ";".join(missing) if missing else "",
            "duplicate_bands": ";".join([f"{b}x{c}" for b, c in dupes.items()]) if dupes else "",
        })
    return pd.DataFrame(rows).sort_values(["mode", "seed"]).reset_index(drop=True)

def _bar_with_error(ax, labels, means, stds, title, ylim=None):
    import numpy as np
    x = np.arange(len(labels))
    means = np.nan_to_num(np.asarray(means, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    stds  = np.nan_to_num(np.asarray(stds,  dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    ax.bar(x, means)
    if np.any(stds > 0):
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="k", capsize=4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.2)

def make_charts(df_long: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in sorted(df_long["mode"].unique().tolist()):
        d = per_band(df_long, mode)
        if not d["accuracy_mean"].notna().any():
            continue

        # Accuracy chart
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar_with_error(ax,
                      d["band"].tolist(),
                      d["accuracy_mean"].tolist(),
                      d["accuracy_std"].tolist(),
                      f"Accuracy by band ({mode})",
                      ylim=(0.0, 1.02))
        fig.tight_layout()
        fig.savefig(outdir / f"accuracy_by_band_{mode}.png", dpi=180)
        plt.close(fig)

        # F1 chart
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar_with_error(ax,
                      d["band"].tolist(),
                      d["f1_mean"].tolist(),
                      d["f1_std"].tolist(),
                      f"F1 by band ({mode})",
                      ylim=(0.0, 1.02))
        fig.tight_layout()
        fig.savefig(outdir / f"f1_by_band_{mode}.png", dpi=180)
        plt.close(fig)

        # Recall chart
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar_with_error(ax,
                      d["band"].tolist(),
                      d["recall_mean"].tolist(),
                      d["recall_std"].tolist(),
                      f"Recall by band ({mode})",
                      ylim=(0.0, 1.02))
        fig.tight_layout()
        fig.savefig(outdir / f"recall_by_band_{mode}.png", dpi=180)
        plt.close(fig)

        # FPR chart
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar_with_error(ax,
                        d["band"].tolist(),
                        d["fpr_mean"].tolist(),
                        d["fpr_std"].tolist(),
                        f"FPR by band ({mode})",
                        ylim=(0.0, 1.02))
        ax.axhline(0.10, linestyle="--", linewidth=1, color="k")
        fig.tight_layout()
        fig.savefig(outdir / f"fpr_by_band_{mode}.png", dpi=180)
        plt.close(fig)

        # ROC-AUC chart
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar_with_error(ax,
                      d["band"].tolist(),
                      d["roc_auc_mean"].tolist(),
                      d["roc_auc_std"].tolist(),
                      f"ROC-AUC by band ({mode})",
                      ylim=(0.0, 1.02))
        fig.tight_layout()
        fig.savefig(outdir / f"roc_auc_by_band_{mode}.png", dpi=180)
        plt.close(fig)

    # Delta chart (stealthy - standard) if both present
    modes = set(df_long["mode"].unique().tolist())
    if {"standard", "stealthy"} <= modes:
        dstd = per_band(df_long, "standard")
        dste = per_band(df_long, "stealthy")
        delta = delta_by_band(dstd, dste).set_index("band")["recall_delta"].reindex(BAND_ORDER)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(len(BAND_ORDER)), delta.values)
        ax.set_xticks(np.arange(len(BAND_ORDER)))
        ax.set_xticklabels(BAND_ORDER, rotation=25, ha="right")
        ax.set_title("Recall delta (stealthy - standard)")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / "delta_recall_stealthy_minus_standard.png", dpi=180)
        plt.close(fig)


def _plot_threshold_ci_forest(ci_long: pd.DataFrame, df_long: pd.DataFrame, metric: str, outdir: Path):
    """
    Plot 95% CI forest by band for a thresholded metric e.g. f1, fpr, precision and more
    Uses rows from threshold_ci_long.csv to help
    """
    if ci_long.empty:
        return
    # Map dataset_name to band using df_long (unique)
    ds_band = df_long[["dataset_name", "band"]].drop_duplicates()
    ci = ci_long.merge(ds_band, on="dataset_name", how="left")
    ci = ci[ci["band"].isin(BAND_ORDER)].copy()

    for mode in sorted(ci["mode"].unique()):
        d = ci[(ci["mode"] == mode) & (ci["metric"] == metric)]
        if d.empty:
            continue
        # Aggregate across seeds *and* datasets within a band: average the bootstrapped means; CI envelopes as mean of bounds
        g = (d.groupby("band")
               .agg(mean=("mean", "mean"),
                    lower_95=("lower_95", "mean"),
                    upper_95=("upper_95", "mean"))
               .reindex(BAND_ORDER))
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(g))
        ax.errorbar(g["mean"], x, xerr=[g["mean"] - g["lower_95"], g["upper_95"] - g["mean"]],
                    fmt="o", capsize=4)
        ax.set_yticks(x)
        ax.set_yticklabels(BAND_ORDER)
        ax.invert_yaxis()
        ttl = f"{metric.upper()} (95% CI) by band — {mode}"
        ax.set_title(ttl)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.2)
        # Draw FPR target line if plotting FPR
        if metric.lower() == "fpr":
            ax.axvline(0.10, linestyle="--", linewidth=1, color="k")
        fig.tight_layout()
        fig.savefig(outdir / f"ci_forest_{metric}_{mode}.png", dpi=180)
        plt.close(fig)

def _plot_auc_ci_forest(df_auc: pd.DataFrame, df_long: pd.DataFrame, which: str, outdir: Path):
    """
    Plots mean +/- 95% CI per band and mode for AUCs using bootstrap_auc_all.csv.
    """
    if df_auc.empty:
        return
    ds_band = df_long[["dataset_name", "band"]].drop_duplicates()
    auc = df_auc.merge(ds_band, on="dataset_name", how="left")
    auc = auc[auc["band"].isin(BAND_ORDER)].copy()

    key_mean = f"{which}_auc_mean"
    key_lo = f"{which}_auc_lower95"
    key_hi = f"{which}_auc_upper95"

    for mode in sorted(auc["mode"].unique()):
        d = auc[auc["mode"] == mode]
        if d.empty:
            continue
        g = (d.groupby("band")
               .agg(mean=(key_mean, "mean"),
                    lower_95=(key_lo, "mean"),
                    upper_95=(key_hi, "mean"))
               .reindex(BAND_ORDER))
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(g))
        ax.errorbar(g["mean"], x, xerr=[g["mean"] - g["lower_95"], g["upper_95"] - g["mean"]],
                    fmt="o", capsize=4)
        ax.set_yticks(x)
        ax.set_yticklabels(BAND_ORDER)
        ax.invert_yaxis()
        ax.set_title(f"{which.upper()}-AUC (95% CI) by band — {mode}")
        ax.set_xlabel("AUC")
        ax.set_xlim(0.0, 1.02)
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"ci_forest_{which}_auc_{mode}.png", dpi=180)
        plt.close(fig)

def _annotate_bars(ax, values):
    for i, v in enumerate(values):
        try:
            ax.text(i, float(v), f"{float(v):.3f}", ha="center", va="bottom", fontsize=8)
        except Exception:
            pass

def _set_ylim_with_errors(ax, means, stds, lower=0.0, pad_frac=0.25, min_pad=0.002):
    """
    Set y-limits using the top of the error bars (mean + std),
    plus padding. Defaults: +25% headroom, at least 0.002 ms.
    """
    m = np.asarray(means, dtype=float)
    s = np.asarray(stds if stds is not None else np.zeros_like(m), dtype=float)
    top = float(np.nanmax(m + s)) if m.size else 0.0
    pad = max(min_pad, pad_frac * max(top, 1e-12))
    ax.set_ylim(lower, top + pad)

def _set_latency_ylim(ax, values, pad_frac=0.25, min_pad=0.002):
    import numpy as np
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    ymax = float(vals.max())
    pad = max(min_pad, pad_frac * ymax)
    ax.set_ylim(0.0, ymax + pad)

def plot_latency_by_seed(lat_by_seed: pd.DataFrame, outdir: Path) -> None:
    """
    Bar charts of per-seed mean latency (ms/sample) across datasets, split by mode.
    Expects columns: ['mode','seed','median_ms_per_sample_mean_across_datasets'].
    """
    if lat_by_seed.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    for mode, grp in lat_by_seed.groupby("mode"):
        d = grp.sort_values("seed")
        x_idx = np.arange(len(d))
        x_labels = d["seed"].astype(int).tolist()
        y_vals = d["median_ms_per_sample_mean_across_datasets"].astype(float).tolist()

        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(x_idx, y_vals)
        _set_latency_ylim(ax, y_vals)
        _annotate_bars(ax, y_vals)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"Latency by seed (ms/sample) — {mode}")
        ax.set_ylabel("ms/sample")
        ax.axhline(10.0, linestyle="--", linewidth=1)  # 10 ms target
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"latency_by_seed_{mode}.png", dpi=180)
        plt.close(fig)

def plot_latency_by_band(lat_all: pd.DataFrame, df_long: pd.DataFrame, outdir: Path) -> None:
    """
    Mean std latency (ms/sample) per band and mode.
    Joins lat_all to df_long to map dataset_name to band.
    Expects lat_all: ['seed','mode','dataset_name','median_ms_per_sample'].
    """
    if lat_all.empty or df_long.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    ds_band = df_long[["dataset_name", "band"]].drop_duplicates()
    merged = lat_all.merge(ds_band, on="dataset_name", how="left")
    merged = merged[merged["band"].isin(BAND_ORDER)].copy()

    for mode, grp in merged.groupby("mode"):
        g = (grp.groupby("band")["median_ms_per_sample"]
                 .agg(["mean", "std"])
                 .reindex(BAND_ORDER))
        fig, ax = plt.subplots(figsize=(10, 3.6))
        means = g["mean"].fillna(0.0).tolist()
        stds  = g["std"].fillna(0.0).tolist()
        _bar_with_error(ax, BAND_ORDER, means, stds, f"Latency by band (ms/sample) — {mode}")
        _set_ylim_with_errors(ax, g["mean"].values, g["std"].values, lower=0.0, pad_frac=0.25, min_pad=0.002)
        ax.axhline(10.0, linestyle="--", linewidth=1)  # 10 ms target
        ax.set_ylabel("ms/sample")
        fig.tight_layout()
        fig.savefig(outdir / f"latency_by_band_{mode}.png", dpi=180)
        plt.close(fig)

def plot_latency_ecdf(lat_all: pd.DataFrame, outdir: Path, sla_ms: float = 10.0,
                      xscale: str = "linear") -> None:
    """
    Draw ECDF of per-(seed,dataset) median ms/sample.
    - Only draw SLA line if it falls inside the zoomed range - p50 and p95.
    """
    import numpy as np
    for mode, grp in lat_all.groupby("mode"):
        vals = np.sort(grp["median_ms_per_sample"].astype(float).values)
        if vals.size == 0:
            continue

        y = np.linspace(0, 1, len(vals), endpoint=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.step(vals, y, where="post")
        ax.set_title(f"Latency ECDF — {mode}")
        ax.set_xlabel("Latency (ms/sample)")
        ax.set_ylabel("Proportion ≤ x")
        ax.grid(True, alpha=0.2)

        if xscale == "log":
            ax.set_xscale("log")
            xmin = max(1e-6, float(vals.min()))
            xmax = float(vals.max())
            pad = max(1e-6, 0.1 * xmax)
            ax.set_xlim(xmin, xmax + pad)
            if xmin <= sla_ms <= xmax + pad:
                ax.axvline(sla_ms, linestyle="--", linewidth=1)
            else:
                ax.annotate(f"SLA {sla_ms:.0f} ms (off-scale)",
                            xy=(xmax, 0.05), ha="right", va="bottom")
        else:
            xmax = float(vals.max())
            pad = max(0.001, 0.10 * xmax)
            ax.set_xlim(0.0, xmax + pad)
            if sla_ms <= xmax + pad:
                ax.axvline(sla_ms, linestyle="--", linewidth=1)
            else:
                ax.annotate(f"SLA {sla_ms:.0f} ms (off-scale)",
                            xy=(xmax, 0.08), ha="right", va="bottom")

        p50, p95 = np.percentile(vals, [50, 95])
        ax.annotate(f"p50={p50:.3f} ms, p95={p95:.3f} ms",
                    xy=(float(vals.max()), 0.02), ha="right", va="bottom")
        fig.tight_layout()
        fig.savefig(outdir / f"latency_ecdf_{mode}.png", dpi=180)
        plt.close(fig)


def run(standard_root: Optional[Path], stealthy_root: Optional[Path], outdir: Path, training_root: Optional[Path] = None) -> None:

    df_std = scan_mode_dir(standard_root, "standard") if standard_root else pd.DataFrame()
    df_ste = scan_mode_dir(stealthy_root, "stealthy") if stealthy_root else pd.DataFrame()

    frames = [d for d in [df_std, df_ste] if not d.empty]
    if not frames:
        print("No data found in the provided roots.")
        return

    df_long = pd.concat(frames, ignore_index=True)
    # Drop NaN for key metrics
    df_long = df_long.dropna(subset=["accuracy", "f1", "recall", "precision", "fpr"])

    # Integrity report + summary
    integ = integrity_report(df_long)
    integ.to_csv(outdir / "integrity_report.csv", index=False)

    summary = {
        "seeds": sorted(df_long["seed"].unique().tolist()),
        "modes": sorted(df_long["mode"].unique().tolist()),
        "bands_present": [b for b in BAND_ORDER if b in set(df_long["band"].unique().tolist())],
        "rows": int(len(df_long)),
        "by_mode_counts": df_long.groupby("mode").size().to_dict(),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Bootstrap AUC CIs (if present in evaluation outputs)
    df_auc_std = scan_bootstrap_auc(standard_root, "standard") if standard_root else pd.DataFrame()
    df_auc_ste = scan_bootstrap_auc(stealthy_root, "stealthy") if stealthy_root else pd.DataFrame()
    auc_frames = [d for d in [df_auc_std, df_auc_ste] if not d.empty]
    if auc_frames:
        df_auc = pd.concat(auc_frames, ignore_index=True)
        df_auc.to_csv(outdir / "bootstrap_auc_all.csv", index=False)
        _plot_auc_ci_forest(df_auc, df_long, which="roc", outdir=outdir)
        _plot_auc_ci_forest(df_auc, df_long, which="pr",  outdir=outdir)

    ci_frames = []
    if not df_std.empty:
        ci_frames.append(scan_threshold_ci(standard_root, "standard"))
    if not df_ste.empty:
        ci_frames.append(scan_threshold_ci(stealthy_root, "stealthy"))
    if ci_frames:
        ci_long = pd.concat(ci_frames, ignore_index=True)
        ci_long.to_csv(outdir / "threshold_ci_long.csv", index=False)

        for metric in ["fpr", "fnr", "f1", "recall"]:
            _plot_threshold_ci_forest(ci_long, df_long, metric, outdir)
        
    # Test latency summaries (median ms/sample only)
    lat_std = scan_latency(standard_root, "standard") if standard_root else pd.DataFrame()
    lat_ste = scan_latency(stealthy_root, "stealthy") if stealthy_root else pd.DataFrame()
    lat_frames = [d for d in [lat_std, lat_ste] if not d.empty]
    if lat_frames:
        lat_all = pd.concat(lat_frames, ignore_index=True)
        lat_all.to_csv(outdir / "test_latency_all.csv", index=False)

        # Per seed x mode: average of dataset medians
        lat_by_seed = (lat_all.groupby(["mode", "seed"])["median_ms_per_sample"]
                              .mean()
                              .reset_index()
                              .rename(columns={"median_ms_per_sample": "median_ms_per_sample_mean_across_datasets"}))
        lat_by_seed.to_csv(outdir / "test_latency_by_seed.csv", index=False)

        # Overall per mode: mean ± std across seeds (using the per-seed means above)
        lat_overall = (lat_by_seed.groupby("mode")["median_ms_per_sample_mean_across_datasets"]
                                  .agg(["mean", "std"])
                                  .reset_index())
        lat_overall.to_csv(outdir / "test_latency_overall.csv", index=False)

        plot_latency_by_seed(lat_by_seed, outdir)
        plot_latency_by_band(lat_all, df_long, outdir)
        plot_latency_ecdf(lat_all, outdir)
    
    df_long.sort_values(["mode", "seed", "band", "dataset_name"], inplace=True)
    (outdir / "evaluation_long_rows.csv").write_text(df_long.to_csv(index=False))

    # Per-band CSVs
    if not df_std.empty:
        per_band(df_long, "standard").to_csv(outdir / "evaluation_per_band_standard.csv", index=False)
    if not df_ste.empty:
        per_band(df_long, "stealthy").to_csv(outdir / "evaluation_per_band_stealthy.csv", index=False)

    # Macro analysis
    macro_seed = macro_by_seed(df_long)
    macro_seed.to_csv(outdir / "evaluation_macro_by_seed_all_modes.csv", index=False)

    if "standard" in set(macro_seed["mode"].unique().tolist()):
        macro_seed[macro_seed["mode"] == "standard"][
            ["seed", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "macro_f1"]
        ].to_csv(outdir / "evaluation_macro_by_seed_standard.csv", index=False)

    if "stealthy" in set(macro_seed["mode"].unique().tolist()):
        macro_seed[macro_seed["mode"] == "stealthy"][
            ["seed", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc", "macro_f1"]
        ].to_csv(outdir / "evaluation_macro_by_seed_stealthy.csv", index=False)

    macro_overall(macro_seed).to_csv(outdir / "evaluation_macro_overall.csv", index=False)

    # Champion analysis: prefer training_root if provided, else uses standard_root
    meta_root = training_root if (training_root and training_root.exists()) else standard_root
    if meta_root:
        meta = build_seed_meta(meta_root)
        if not meta.empty:
            merged = macro_seed.merge(meta, on="seed", how="left")
            merged.to_csv(outdir / "evaluation_macro_by_seed_with_meta.csv", index=False)

            # Champion counts
            (merged.groupby("champion")
                   .size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False)
                   .to_csv(outdir / "evaluation_champion_counts.csv", index=False))

            # Performance by champion
            (merged.groupby(["mode", "champion"])
                   .agg(accuracy_mean=("accuracy", "mean"),
                        f1_mean=("f1", "mean"),
                        recall_mean=("recall", "mean"),
                        roc_auc_mean=("roc_auc", "mean"),
                        n=("seed", "nunique"))
                   .reset_index()
                   .to_csv(outdir / "evaluation_perf_by_champion.csv", index=False))

            meta.to_csv(outdir / "evaluation_champion_latency_by_seed.csv", index=False)

    # Delta analysis
    modes = set(df_long["mode"].unique().tolist())
    if {"standard", "stealthy"} <= modes:
        std_band = per_band(df_long, "standard")
        ste_band = per_band(df_long, "stealthy")
        delta_by_band(std_band, ste_band).to_csv(outdir / "evaluation_delta_by_band.csv", index=False)

    # Charts
    make_charts(df_long, outdir)

    # Console summary
    print("Aggregation complete.")
    print(" Seeds:", sorted(df_long["seed"].unique().tolist()))
    print(" Modes:", sorted(df_long["mode"].unique().tolist()))
    print(" Bands:", [b for b in BAND_ORDER if b in set(df_long["band"].unique().tolist())])
    print(" Outputs written to:", outdir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=str, default=None, help="Optional: path to TRAINING outputs (seed_*/experiment_results.json) for champion/meta, Or else uses --standard-root.")
    ap.add_argument("--standard-root", type=str, default=None,
                    help="Path to folder containing seed_* with STANDARD thresholds.")
    ap.add_argument("--stealthy-root", type=str, default=None,
                    help="Path to folder containing seed_* with STEALTHY thresholds.")
    ap.add_argument("--outdir", type=str, required=True,
                    help="Output directory for CSV/PNG artefacts.")
    args = ap.parse_args()

    std_root = Path(args.standard_root) if args.standard_root else None
    ste_root = Path(args.stealthy_root) if args.stealthy_root else None
    outdir = Path(args.outdir)
    train_root = Path(args.training_root) if args.training_root else None

    if std_root is None and ste_root is None:
        print("Provide at least one of --standard-root or --stealthy-root.")
        return

    run(std_root, ste_root, outdir, training_root=train_root)

if __name__ == "__main__":
    main()