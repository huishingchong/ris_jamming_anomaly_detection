#!/usr/bin/env python3
"""
Comprehensive analysis and visualisation of feature distributions across 
Normal, RIS Jamming, and Active Jamming scenarios.

This script:
1. Loads arbitrary CSV dataset with feature columns and labels
2. Identifies most discriminative features using statistical tests
3. Generates plots like correlation matrix and circular importance diagram

Author: Hui Shing
"""

import argparse
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, f_oneway
import textwrap
from typing import Dict, List, Tuple
from dataclasses import dataclass
from matplotlib import cm, colormaps, colors, patheffects as pe

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.format': 'png'
})


@dataclass
class FeatureStats:
    """Statistical analysis results for a feature across classes."""
    feature_name: str
    f_statistic: float
    f_p_value: float
    kruskal_statistic: float
    kruskal_p_value: float
    class_means: Dict[str, float]
    class_stds: Dict[str, float]
    effect_size: float
    discrimination_score: float


class FeatureAnalyser:    
    def __init__(self, random_state: int = 42, alpha: float = 0.05):
        """
        Args:
            random_state: Random seed for reproducibility
            alpha: Significance level for statistical tests
        """
        self.random_state = random_state
        self.alpha = alpha
        self.class_colors = {
            'Normal': '#2E8B57',
            'RIS Jamming': '#DC143C',
            'Active Jamming': '#FF8C00',
            'Stealthy': '#8B008B'
        }
        
        np.random.seed(random_state)
    
    def load_and_validate_data(self, csv_path: Path, specific_features: List[str] = None, binary_mode: bool = False, stealthy_mode: bool = False) -> pd.DataFrame:
        """
        Args:
            csv_path: Path to CSV file
            specific_features: List of specific features to analyse
            binary_mode: If True, filter out Active (label=2) for binary analysis
            stealthy_mode: If True, filter to only normal and stealthy band samples
        Returns: Validated DataFrame with features and labels
        """
        print(f"Loading dataset from {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples with {df.shape[1]} columns")
        
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain 'label' column")
        
        # Handle stealthy mode filtering
        if stealthy_mode:
            if 'band_name' not in df.columns:
                raise ValueError("Stealthy mode requires 'band_name' column in dataset")
            
            before = len(df)
            # Keep normal samples (label=0) and stealthy jamming samples
            stealthy_mask = (df['band_name'] == 'stealthy') & (df['label'] == 1)
            normal_mask = df['label'] == 0
            df = df[stealthy_mask | normal_mask].copy()
            after = len(df)
            
            print(f"Stealthy mode: filtered to normal + stealthy samples. Kept {after}/{before} rows")
            
            # Set class names for stealthy analysis
            self.class_names = {0: 'Normal', 1: 'Stealthy'}
            
        elif binary_mode:
            before = len(df)
            df = df[df['label'].isin([0, 1])].copy()
            after = len(df)
            print(f"Binary mode: filtered out Active (2). Kept {after}/{before} rows (labels in {{0,1}})")
            self.class_names = {0: 'Normal', 1: 'RIS Jamming'}
        else:
            self.class_names = {0: 'Normal', 1: 'RIS Jamming', 2: 'Active Jamming'}
        
        unique_labels = sorted(df['label'].unique())
        print(f"Found labels: {unique_labels}")
        
        # Update class names based on available labels
        available_classes = {}
        for label in unique_labels:
            if label in self.class_names:
                available_classes[self.class_names[label]] = label
            else:
                available_classes[f'Class_{label}'] = label
        
        self.available_classes = available_classes
        print(f"Class mapping: {available_classes}")
        
        # Extract feature columns (exclude metadata and ensure it is a numeric)
        exclude_cols = ['label', 'band_name']
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        print(f"Found {len(feature_cols)} numeric feature columns")
        
        # Apply specific feature selection if provided
        if specific_features:
            available_specific = [f for f in specific_features if f in feature_cols]
            missing_specific = [f for f in specific_features if f not in feature_cols]
            
            if missing_specific:
                print(f"Warning: Requested features not found: {missing_specific}")
                print(f"Available features: {sorted(feature_cols)}")
            
            if not available_specific:
                raise ValueError(f"None of the specified features found in dataset. Available: {sorted(feature_cols)}")
            
            feature_cols = available_specific
            print(f"Using {len(feature_cols)} specified features: {feature_cols}")
        
        # Filter out features if they have missing values or constan values
        valid_features = []
        for col in feature_cols:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.1:
                print(f"Skipping {col}: {missing_pct:.1%} missing values")
                continue
            
            if df[col].nunique() <= 1:
                print(f"Skipping {col}: constant values")
                continue
            
            valid_features.append(col)
        
        if not valid_features:
            raise ValueError("No valid features remain after filtering")
        
        print(f"Using {len(valid_features)} valid features")
        
        # Return clean dataset (though exclude band_name from features but keep for filtering)
        clean_df = df[valid_features + ['label']].dropna()
        print(f"Final dataset: {len(clean_df)} samples, {len(valid_features)} features")
        
        return clean_df
    
    def calculate_feature_statistics(self, df: pd.DataFrame) -> Tuple[List[FeatureStats], pd.DataFrame]:
        """
        Args:
            df: DataFrame with features and labels
        Returns: Tuple of (feature statistics list, ranking DataFrame)
        """
        feature_cols = [col for col in df.columns if col != 'label']
        feature_stats = []
                
        # Group data by label for analysis
        groups = {lbl: df[df['label'] == lbl][feature_cols].values for lbl in sorted(df['label'].unique())}
        
        ranking_rows = []
        for j, feat in enumerate(feature_cols):
            # Collect this feature across groups
            samples = [groups[g][:, j] for g in groups]
            
            # Guard against degenerate groups
            if any(len(s) < 2 for s in samples):
                F, p = np.nan, np.nan
                eta2 = np.nan
                kruskal_stat, kruskal_p = np.nan, np.nan
            else:
                F, p = f_oneway(*samples)
                
                # Calculate eta squared (effect size): SSB / SST
                all_vals = df[feat].values
                grand_mean = np.mean(all_vals)
                ss_total = np.sum((all_vals - grand_mean)**2)
                ss_between = 0.0
                for g in groups:
                    vals = df.loc[df['label'] == g, feat].values
                    ss_between += len(vals) * (np.mean(vals) - grand_mean)**2
                eta2 = ss_between / ss_total if ss_total > 0 else np.nan
                
                # Kruskal-Wallis test (non-parametric alternative)
                try:
                    kruskal_stat, kruskal_p = kruskal(*samples)
                except ValueError:
                    kruskal_stat, kruskal_p = 0.0, 1.0
            
            # Prepare class statistics
            class_means = {}
            class_stds = {}
            for class_name, label_value in self.available_classes.items():
                mask = df['label'] == label_value
                data = df[mask][feat].values
                class_means[class_name] = np.mean(data)
                class_stds[class_name] = np.std(data)
            
            # Discrimination score (combines statistical significance and effect size)
            discrimination_score = eta2 * (1 - min(p, 1.0)) if not np.isnan(eta2) and not np.isnan(p) else 0.0
            
            stats_obj = FeatureStats(
                feature_name=feat,
                f_statistic=float(F) if np.isfinite(F) else np.nan,
                f_p_value=float(p) if np.isfinite(p) else np.nan,
                kruskal_statistic=float(kruskal_stat) if np.isfinite(kruskal_stat) else np.nan,
                kruskal_p_value=float(kruskal_p) if np.isfinite(kruskal_p) else np.nan,
                class_means=class_means,
                class_stds=class_stds,
                effect_size=float(eta2) if np.isfinite(eta2) else np.nan,
                discrimination_score=discrimination_score
            )
            
            feature_stats.append(stats_obj)
            
            ranking_rows.append({
                "feature": feat,
                "F_value": float(F) if np.isfinite(F) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "eta_squared": float(eta2) if np.isfinite(eta2) else np.nan,
                "kruskal_statistic": float(kruskal_stat) if np.isfinite(kruskal_stat) else np.nan,
                "kruskal_p_value": float(kruskal_p) if np.isfinite(kruskal_p) else np.nan
            })
        
        # Sort by discrimination score (descending)
        feature_stats.sort(key=lambda x: x.discrimination_score, reverse=True)
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame(ranking_rows)
        ranking_df = ranking_df.sort_values(["eta_squared", "F_value"], ascending=[False, False]).reset_index(drop=True)
        
        # Print full ranking
        print(" Full discriminative ranking (all features) ")
        for i, row in ranking_df.iterrows():
            print(f"{i+1:3d}. {row['feature']}: F={row['F_value']:.2f}  p={row['p_value']:.3e}  η²={row['eta_squared']:.3f}")
        
        return feature_stats, ranking_df
    
    def create_distribution_plots(self, df: pd.DataFrame, features: List[str], output_dir: Path):
        """
        Create comprehensive distribution plots for selected features.
        
        Args:
            df: DataFrame with features and labels
            features: List of feature names to plot
            output_dir: Directory to save plots
        """
        print(f"Creating distribution plots for {len(features)} features")
        
        # Debug: Check feature value ranges
        for feature in features:
            feature_values = df[feature]
            print(f"Feature {feature}: min={feature_values.min():.2e}, max={feature_values.max():.2e}, mean={feature_values.mean():.2e}")
            
            # Special check for spectral_centroid
            if 'centroid' in feature.lower():
                extreme_count = len(feature_values[feature_values > 1e5])  # > 100kHz
                if extreme_count > 0:
                    print(f"Spectral centroid extreme values (>100kHz): {extreme_count}/{len(feature_values)} samples")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for plotting with class names
        df_plot = df.copy()
        df_plot["class_name"] = df_plot["label"].map(self.class_names)
        order = [self.class_names[k] for k in sorted(self.class_names.keys()) if k in df['label'].unique()]
        
        # Create individual feature distribution plots
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes_flat = [axes]
        elif n_rows == 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes_flat[idx]
            
            for class_name in order:
                label_value = self.available_classes[class_name]
                data = df[df['label'] == label_value][feature]
                
                # Histogram with transparency
                ax.hist(data, bins=30, alpha=0.6, label=class_name, 
                       color=self.class_colors.get(class_name, '#333333'), density=True)
                
                # KDE overlay
                if len(data) > 1:
                    data_clean = data.dropna()
                    if len(data_clean) > 1 and data_clean.std() > 0:
                        sns.kdeplot(data=data_clean, ax=ax, 
                                  color=self.class_colors.get(class_name, '#333333'), 
                                  linewidth=2, alpha=0.8)
            
            ax.set_title(f'Distribution: {feature}', fontweight='bold')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_distributions.png')
        plt.close()
        
        # Create box plots for statistical comparison
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_features == 1:
            axes_flat = [axes]
        elif n_rows == 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes_flat[idx]
            
            plot_data = []
            plot_labels = []
            
            for class_name in order:
                label_value = self.available_classes[class_name]
                data = df[df['label'] == label_value][feature].dropna()
                plot_data.append(data)
                plot_labels.append(class_name)
            
            box_plot = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)
            
            for patch, class_name in zip(box_plot['boxes'], plot_labels):
                patch.set_facecolor(self.class_colors.get(class_name, '#333333'))
                patch.set_alpha(0.7)
            
            ax.set_title(f'Box Plot: {feature}', fontweight='bold')
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels if needed
            if max(len(label) for label in plot_labels) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_boxplots.png')
        plt.close()
        
        print("Distribution plots saved successfully")
    
    def create_correlation_analysis(self, df: pd.DataFrame, features: List[str], output_dir: Path):
        """
        Creating a feature correlation heatmap
        Args:
            df: DataFrame with features and labels  
            features: List of feature names to analyse
            output_dir: Directory to save plots
        """
        feature_df = df[features]
        correlation_matrix = feature_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_correlations.png')
        plt.close()
        
        print("Correlation analysis saved successfully")
    
    def create_discriminative_analysis(self, df: pd.DataFrame, feature_stats: List[FeatureStats], output_dir: Path):
        """
        Create plots per feature for discriminative feature analysis
        
        Args:
            df: DataFrame with features and labels
            feature_stats: List of feature statistics
            output_dir: Directory to save plots
        """        
        top_features = feature_stats[:min(15, len(feature_stats))]
        
        plt.figure(figsize=(12, 8))
        feature_names = [stat.feature_name for stat in top_features]
        discrimination_scores = [stat.discrimination_score for stat in top_features]
        
        bars = plt.barh(range(len(feature_names)), discrimination_scores)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Discrimination Score (Effect Size × (1 - p-value))')
        plt.title('Feature Discriminative Power Ranking', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Colour the bars by discrimination level
        max_score = max(discrimination_scores) if discrimination_scores else 1.0
        colors_list = plt.cm.RdYlGn([score/max_score for score in discrimination_scores])
        for bar, color in zip(bars, colors_list):
            bar.set_color(color)
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'discriminative_power_ranking.png')
        plt.close()
        
        print("Discriminative analysis saved successfully")
    
    def create_circular_importance_plot(self, feature_stats: List[FeatureStats], output_dir: Path, 
                                   min_eta_threshold: float = 0.00, top_k: int = None, 
                                   highlight_features: List[str] = None):
        """Create a nice circular importance plot."""
        
        # Filter and sort features
        valid = [s for s in feature_stats
                if s.effect_size is not None and not np.isnan(s.effect_size)
                and s.effect_size >= min_eta_threshold]
        if not valid:
            print("Warning: No features meet minimum eta-squared threshold for circular plot")
            return

        stats = sorted(valid, key=lambda x: x.effect_size, reverse=True)
        if top_k is not None and top_k < len(stats):
            stats = stats[:top_k]

        feats = [s.feature_name for s in stats]
        eta2  = np.array([float(s.effect_size) for s in stats], dtype=float)
        n = len(feats)
        if n == 0:
            print("Warning: No features to plot after filtering")
            return

        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        width  = (2*np.pi / n) * 0.80

        gamma = 0.70
        emax = max(eta2.max(), 1e-9)
        r_inner = 0.22
        r_maxbar = 0.86
        r_bars = r_inner + (eta2 / emax)**gamma * (r_maxbar - r_inner)

        # Label rings - avoid overlap
        r_label_1 = 0.99
        r_label_2 = 1.06
        ylim_max = 1.14

        # Angle of seperation
        min_sep_deg = 15
        min_sep = np.deg2rad(min_sep_deg)

        # Need to revise this bit
        # Decide ring per label to avoid text collision???
        ring_choice = []
        last_angle = None
        for a in angles:
            if last_angle is None or abs((a - last_angle + np.pi) % (2*np.pi) - np.pi) >= min_sep:
                ring_choice.append(1)
            else:
                ring_choice.append(2)
            last_angle = a

        fig, ax = plt.subplots(figsize=(11.5, 11.5), subplot_kw={'projection': 'polar'}, dpi=300)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, ylim_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, alpha=0.18, linewidth=0.8)
        ax.spines['polar'].set_alpha(0.25)
        ax.set_facecolor('white')

        # Create the bars
        cmap = colormaps.get_cmap('Spectral')
        norm = colors.Normalize(vmin=float(eta2.min()), vmax=float(emax))
        bar_colors = cmap(norm(eta2))
        bars = ax.bar(angles, r_bars, width=width, bottom=0.0,
                    color=bar_colors, edgecolor='white', linewidth=1.2, alpha=0.97)

        # Add the labels intoo
        for ang, r, val in zip(angles, r_bars, eta2):
            r_mid = max(r_inner + 0.06, r * 0.58)
            t = ax.text(ang, r_mid, f'{val:.3f}',
                        ha='center', va='center',
                        fontsize=15.5, fontweight='bold', color='white')
            t.set_path_effects([pe.withStroke(linewidth=2.2, foreground='black', alpha=0.55)])

        line_color = '#b0b4bb'
        line_lw = 1.1

        for ang, r_tip, name, ring in zip(angles, r_bars, feats, ring_choice):
            r_lab = r_label_1 if ring == 1 else r_label_2

            ax.plot([ang, ang], [r_tip + 0.010, r_lab - 0.012],
                    color=line_color, linewidth=line_lw, alpha=0.95, solid_capstyle='round', clip_on=False)

            deg = np.degrees(ang)
            ha = 'right' if 90 < deg < 270 else 'left'

            clean = name.replace('_', ' ')
            if len(clean) > 26:
                clean = textwrap.fill(clean, 26)

            ax.text(ang, r_lab, clean,
                    rotation=0, rotation_mode='anchor',
                    ha=ha, va='center', fontsize=14, fontweight='semibold', color='black')

        ax.set_title('Circular Feature Importance (η²)', pad=18, fontsize=22, fontweight='bold')
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / 'circular_feature_importance.png'
        fig.tight_layout()
        fig.savefig(out, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Circular importance plot saved at {out}")
    
    def generate_summary_report(self, df: pd.DataFrame, feature_stats: List[FeatureStats], 
        ranking_df: pd.DataFrame, output_dir: Path, binary_mode: bool = False, stealthy_mode: bool = False):
        """
        Args:
            df: DataFrame with features and labels
            feature_stats: List of feature statistics
            ranking_df: DataFrame with full feature rankings
            output_dir: Directory to save report
            binary_mode: Whether analysis was run in binary mode
            stealthy_mode: Whether analysis was run in stealthy mode
        """        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FEATURE DISTRIBUTION ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Analysis mode
        if stealthy_mode:
            mode_str = "Stealthy Analysis (Normal vs Stealthy Jamming)"
        elif binary_mode:
            mode_str = "Binary Classification (Normal vs RIS Jamming)"
        else:
            mode_str = "Multi-class Classification"
        report_lines.append(f"Analysis Mode: {mode_str}")
        report_lines.append("")
        
        # Dataset summary
        report_lines.append("DATASET SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total samples: {len(df):,}")
        report_lines.append(f"Features analysed: {len([col for col in df.columns if col != 'label'])}")
        report_lines.append("")
        
        # Class distribution
        report_lines.append("CLASS DISTRIBUTION")
        report_lines.append("-" * 40)
        for class_name, label_value in self.available_classes.items():
            count = len(df[df['label'] == label_value])
            percentage = count / len(df) * 100
            report_lines.append(f"{class_name}: {count:,} samples ({percentage:.1f}%)")
        report_lines.append("")
        
        # Top discriminative features
        report_lines.append("TOP DISCRIMINATIVE FEATURES")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Rank':<4} {'Feature':<25} {'F-stat':<10} {'p-value':<12} {'η²':<8} {'Discrim':<8}")
        report_lines.append("-" * 67)
        
        for rank, stat in enumerate(feature_stats[:15], 1):
            p_str = f"{stat.f_p_value:.2e}" if stat.f_p_value < 0.001 else f"{stat.f_p_value:.4f}"
            report_lines.append(f"{rank:<4} {stat.feature_name:<25} {stat.f_statistic:<10.2f} "
                              f"{p_str:<12} {stat.effect_size:<8.3f} {stat.discrimination_score:<8.3f}")
        report_lines.append("")
        
        report_lines.append("FEATURE MEANS BY CLASS (Top 8 Features)")
        report_lines.append("-" * 40)
        
        for stat in feature_stats[:8]:
            report_lines.append(f"\n{stat.feature_name}:")
            for class_name in self.available_classes.keys():
                mean_val = stat.class_means[class_name]
                std_val = stat.class_stds[class_name]
                report_lines.append(f"  {class_name}: {mean_val:.4f} ± {std_val:.4f}")
        
        report_lines.append("")
        
        report_lines.append("-" * 40)
        top_5_features = [stat.feature_name for stat in feature_stats[:5]]
        top_10_features = [stat.feature_name for stat in feature_stats[:10]]
        
        report_lines.append("Top 5 features:")
        report_lines.append(f"  {' '.join(top_5_features)}")
        report_lines.append("")
        report_lines.append("Top 10 features:")
        report_lines.append(f"  {' '.join(top_10_features)}")
        report_lines.append("")
        
        report_lines.append("=" * 80)
        
        with open(output_dir / 'analysis_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        # Save full feature ranking CSV
        ranking_df.to_csv(output_dir / 'feature_ranking_full.csv', index=False)
        print(f"Saved full ranking to {output_dir / 'feature_ranking_full.csv'}")
        
        print("Summary report generated successfully")
    
    def run_analysis(self, csv_path: Path, output_dir: Path, top_n_features: int = 8, 
        specific_features: List[str] = None,
        binary_mode: bool = False, stealthy_mode: bool = False):
        """
        Run a complete feature analysis pipeline.
        Args:
            csv_path: Path to CSV dataset
            output_dir: Directory to save results
            top_n_features: Number of top features to analyse in detail
            specific_features: List of specific features to analyse
            binary_mode: Whether to run binary classification (Normal vs RIS only)
            stealthy_mode: Whether to run stealthy analysis (Normal vs Stealthy only)
        """
        print(" Starting Feature Distribution Analysis ")
        
        # Load and validate data
        df = self.load_and_validate_data(csv_path, specific_features, binary_mode, stealthy_mode)
        
        # Calculate feature statistics
        feature_stats, ranking_df = self.calculate_feature_statistics(df)
        
        # Select features for detailed analysis
        if specific_features:
            top_features = [f for f in specific_features if f in [stat.feature_name for stat in feature_stats]]
        else:
            top_features = [stat.feature_name for stat in feature_stats[:top_n_features]]
        all_features = [f for f in df]
        print(f"Selected {len(top_features)} features for detailed analysis: {top_features}")
        
        self.create_distribution_plots(df, top_features, output_dir)
        self.create_correlation_analysis(df, all_features, output_dir)
        self.create_discriminative_analysis(df, feature_stats, output_dir)
        self.create_circular_importance_plot(feature_stats, output_dir)

        self.generate_summary_report(df, feature_stats, ranking_df, output_dir, binary_mode, stealthy_mode)
        
        print("Feature Analysis Complete")
        print(f"Results saved to: {output_dir}")
        
        return {
            'feature_stats': feature_stats,
            'top_features': top_features,
            'ranking_df': ranking_df,
            'dataset_summary': {
                'n_samples': len(df),
                'n_features': len([col for col in df.columns if col != 'label']),
                'classes': self.available_classes,
                'binary_mode': binary_mode,
                'stealthy_mode': stealthy_mode
            }
        }
        
def main():
    parser = argparse.ArgumentParser(
        description='Feature Distribution Analysis for RIS Jamming Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Standard multiclass analysis
            python3 feature_analysis.py --csv dataset.csv --output results/analysis
            
            # Binary classification (Normal vs RIS only)
            python3 feature_analysis.py --csv dataset.csv --binary --output results/binary_analysis
            
            # Focus on specific features
            python3 feature_analysis.py --csv dataset.csv --features sinr_estimate mean_psd_db spectral_entropy --output results/analysis
        """
    )
    
    parser.add_argument('--csv', required=True, help='Path to CSV dataset file')
    parser.add_argument('--output', default='results/feature_analysis', 
                       help='Output directory for results (default: results/feature_analysis)')
    parser.add_argument('--binary', action='store_true',
                       help='If set then analyse only Normal (0) vs RIS Jamming (1), excludes Active (2)')
    parser.add_argument('--stealthy', action='store_true',
                       help='If set then analyse only Normal (0) vs Stealthy band samples - requires band_name column')
    parser.add_argument('--features', nargs='*', 
                       help='Specific features to analyse (space-separated list). If not specified, all valid features are used.')
    parser.add_argument('--top-features', type=int, default=10,
                       help='Number of top discriminative features to analyse in detail (default: 10, ignored if --features is used)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.binary and args.stealthy:
        print("Note: Using both --binary and --stealthy modes. Stealthy mode takes precedence.")
        print("Analysis will focus on Normal vs Stealthy band samples only.")
    
    # Setup
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1
    
    
    # Run the analysis
    try:
        analyser = FeatureAnalyser(random_state=args.random_seed)
        results = analyser.run_analysis(
            csv_path, 
            output_dir, 
            args.top_features,
            specific_features=args.features,
            binary_mode=args.binary,
            stealthy_mode=args.stealthy
        )
        
        print("\n" + "="*60)
        print("FEATURE ANALYSIS SUMMARY")
        print("="*60)
        print(f"Dataset: {results['dataset_summary']['n_samples']:,} samples, "
              f"{results['dataset_summary']['n_features']} features")
        print(f"Classes: {list(results['dataset_summary']['classes'].keys())}")
        
        if args.stealthy:
            mode_str = "Stealthy (Normal vs Stealthy)"
        elif args.binary:
            mode_str = "Binary (Normal vs RIS)"
        else:
            mode_str = "Multiclass"
        print(f"Analysis mode: {mode_str}")
        
        if args.features:
            print(f"\nSpecified features analysed: {len(results['top_features'])}")
            for i, feature in enumerate(results['top_features'], 1):
                stat = next(s for s in results['feature_stats'] if s.feature_name == feature)
                print(f"  {i:2d}. {feature} (η²={stat.effect_size:.3f})")
        else:
            print(f"\nTop {len(results['top_features'])} discriminative features:")
            for i, feature in enumerate(results['top_features'], 1):
                stat = results['feature_stats'][i-1]
                print(f"  {i:2d}. {feature} (η²={stat.effect_size:.3f})")
        
        
        print(f"\nResults saved to: {output_dir}")
        print("Generated files:")
        print("feature_distributions.png - Distribution plots")
        print("feature_boxplots.png - Box plot comparisons")  
        print("feature_correlations.png - Correlation matrix")
        print("discriminative_power_ranking.png - Feature rankings")
        print("circular_feature_importance.png - Circular importance plot")
        print("feature_ranking_full.csv - Complete feature ranking data")
        print("analysis_report.txt")
        
        # Print top features for easy copy-paste into other scripts
        print(f"\n{'='*60}")
        print("TOP DISCRIMINATIVE FEATURES FOR COPY-PASTE:")
        print("="*60)
        if args.features:
            features_for_ml = ' '.join(results['top_features'])
            print(f"Specified features: {features_for_ml}")
        else:
            top_5_features = ' '.join(results['top_features'][:5])
            all_analysed_features = ' '.join(results['top_features'])
            print(f"Top 5 features: {top_5_features}")
            print(f"All analysed features: {all_analysed_features}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())