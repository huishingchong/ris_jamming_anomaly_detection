# ris_jamming_anomaly_detection

Author: Hui Shing Chong

Date: September 2025

Submitted in partial fulfillment of the requirements for the degree in Computing (Security & Reliability) MSc of Imperial College London

## Summary

This repository contains the source code for Dissertation 'Machine Learning Anomaly Detection for Illegitimate RIS-based Jamming Attacks'. We explore how we can use ML models to detect RIS-based jamming, an attack that causes destructive interference on the receiver.

## Quick Start

### Prerequisites

- Python 3.11
- MATLAB R2025a (recommended)
- 4GB+ free disk space

## Project Structure
```
ris_jamming_anomaly_detection/
├── datasets/
│   ├── jamming_features_research_train  # CSV test datasets
│   └── jamming_features_research_moderate_test.csv
      ...
├── experimental_results/
│   ├── rq2/  # supervised learning model artifacts and JSON
│   ├── eval/                 # standard-policy JSON/CSV
│   ├── eval_stealthy_threshold/ # stealthy-policy JSON/CSV
│   └── feature_analysis/     # generated plots
├── figures/                  # paper-ready figures copied from experimental_results/figures
├── matlab/                   # MATLAB validation, simulation, signal generation scripts
│   ├── signals/ # store raw signal files .mat
│   ├── pre_generated_validation_figures/ # Validation plots
│   ├── validation_script.m
│   ├── ris_vs_active_jamming_signal_comparison.m
│   ├── solve_ris_jamming_optimisation.m
│   ├── generate_raw_signals_stratified.m
│   ├── extract_features_research.m
│   ├── generate_features_research.m
│   ├── (...other pngs and plots produced from MATLAB for report)
├── src/  # Helper classes for ML pipeline
│   ├── data_handler/
│   ├── models/
│   ├── features/
│   ├── utils/
│   ├── timing/
│   └── __init__.py
├── aggregate_seeds.py
├── evaluation.py
├── feature_analysis.py
├── supervised_detection.py
├── requirements.txt
├── README.md
```
### MATLAB

MATLAB is used during the implementation of Lyu et al's [1] RIS-based jamming algorithm and its validation. The scripts are .m files which can only be run on MATLAB. For our ML pipeline, MATLAB scripts are also used to generate raw signals for training and test (stored in .mat), then converting them to datasets (CSVs) for ML detection.

The MATLAB (.m) files are provided so you can generate your own signals and ML datasets with custom configurations, just keep in mind the input/output directory configuration which is usually at the top of the scripts. You don't need to rerun the generation scripts as all the CSV files (ML datasets) are provided here. .mat files (raw signals) are not able to fit therefore will be provided in a Google Drive. They can be run directly with python scripts provided the input path and file name is configured correctly.

### Python

Python scripts are used for ML training, evaluation and Experimental Results.

## Installation

1. **Clone or download the project**

   ```bash
   git clone <repository-url>
   cd ris_jamming_anomaly_detection
   ```

2. Create virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```

3. You can run MATLAB scripts by opening directory on MATLAB software, usually calling the script on the command window.

- On MATLAB, at the top where you can configure your folder path, ensure you are in the local copy of this directory (cloned this repository) and inside the `matlab` folder in this repository, alternatively, you can use the Command Window `cd matlab`. You should see that you can open the .m files on the left.
- MATLAB beginner-friendly guide here: https://matlabacademy.mathworks.com/details/matlab-onramp/gettingstarted

4. Run python scripts in the current directory to execute the training and evaluation of models for detecting RIS-based jamming attacks.

For both MATLAB and Python scripts, consult the `File/Folder Explanations` section below for helpful descriptions on how to run each script and what to expect.

## Dependencies

### CVX - Convex Optimization Toolbox

To run the RIS-based jamming simulation (e.g. validation_script.m, generate_raw_signals_stratified.m) on MATLAB, CVX is needed, which is a tool used for convex optimisation, as part of the algorithm described by Lyu et al. [1]

1. To download: https://github.com/cvxr/CVX/releases (get the version 2.2)
2. Extract CVX into a directory e.g. ~/MATLAB/cvx
3. In MATLAB (command window), you will need to navigate into the CVX folder e.g. `cd ~/MATLAB/cvx`
4. Next you need to run `cvx_setup`
5. This registers CVX with MATLAB. To further verify the installation: Run `cvx_version`, you should see Version 2.2 listed.

- **Note:** if you move the cvx folder you need to follow steps 3-5 again. You don't need to run `cvx_setup` all the time, its connection usually persist throughout MATLAB session. If the RIS jamming optimisation script doesn't work as intended, you may need to do Steps 3-5.

- **Note for mac users:** CVX installation may require allowing unverified applications through System Preferences > Security \& Privacy due to MacOS protection policies. Helpful guide: https://macpaw.com/how-to/fix-macos-cannot-verify-that-app-is-free-from-malware

### MATLAB

MATLAB R2025a is recommended, and the following toolboxes is mandatory:
Signal processing Toolbox, DSP, Statistics Toolbox, Communications Toolbox

### Other

Other important libraries used is scikit-learn for ML pipeline, matplotlib for plots and visualisation.

## File/Folder Explanations

### MATLAB

solve_ris_jamming_optimisation.m

- Algorithm implementation RIS-based jamming, refer to Lyu et al. [1] for more details
- Not standalone to run, instead used as a helper script called by validation_script.m, ris_vs_active_jamming_signal_comparison.m, generate_raw_signals_stratified.m

validation_script.m

- Produces Figures 2-5 equivalent from Lyu et al.'s Performance Evaluation Section [1] Refer to Validation chapter for the graphs
- Standalone MATLAB script that can be run in command window `validation_script`, feel free change beamforming vector (omega) as what was done during my Beamforming Investigation
- Note that this takes a long time to complete as we are aggregating over statistical runs per plot for robust validation

ris_vs_active_jamming_signal_comparison.m

- Script to explore the potential distinctive signal characteristics produced by RIS vs conventional active jamming
- Compares a RIS jamming and an active jamming signal which achieve the same attack effectiveness (or similar depending on run) and provide power, spectral, etc. visualisations for comparison
- Produces the "RIS vs Active Jamming: Signal Characteristics for Feature Engineering" figure seed in the report
- Standalone MATLAB script that can be run in command window `ris_vs_active_jamming_signal_comparison`

generate_raw_signals_stratified.m

- Generate raw signals which are stored in .mat, for later feature extraction
- This was used to generate training data and test sets
- Run `generate_raw_signals_stratified` directly in command window, you can also modify configuration parameters to customise generation

generate_features_dataset_research.m

- Generates datasets ready for ML training or testing, for experimental results
- Run `generate_features_dataset_research` directly in command window, but ensure correct input path to the .mat file and note the output directory
- Raw signal files used for this research are in the signal folder (contains the test set).
- Input: .mat file, Output: .csv file

extract_features_dataset.m

- Helper function to extract and compute features from signals, used by generate_features_dataset_research.m
- Not standalone script to run

Other helper standalone scripts: distribution_analysis.m, explore_mat_file.m which I used for checks

### Python

To reproduce my Experimental Results:
feature_analysis.py

```
python3 feature_analysis.py --csv datasets/jamming_features_research.csv --binary --output experimental_results/feature_analysis

```

supervised_detection.py

- Load dataset (train/val/test splits), prepare features and train supervised learning models
- Per seed, creates model_artifacts.joblib and experiment_results.json containing the configuration and model & threshold selection, and val/test metrics

```python3 supervised_detection.py \
  --csv datasets/train_jamming_features_research.csv \
  --output results/rq2 \
  --seeds 42 123 456 789 999 \
  --standard-target-fpr 0.10 \
  --standard-guard 0.02 \
  --stealthy-target-fnr 0.10 \
  --stealthy-guard 0.02 \
  --stealthy-weight 2.0 \
  --calibration-method isotonic --bootstrap-iters 1000 --enable-tuning
```

- The output path is where you will store model artifacts and json record of the training
- `--enable-tuning` turns on model hyperparameter tuning which might take a while, to disable, simply run without the flag

evaluation.py

- Run models across the seeds against test datasets, collects metrics extensively for reporting
- Runs models against test CSVs and output performance results: evaluation_complete_results.json

- For evaluating model on standard threshold:

```
for seed in 42 123 456 789 999; do python3 evaluation.py --model results/rq2/seed_${seed}/model_artifacts.joblib --bootstrap-iters 1000 --test-csvs datasets/jamming_features_research_stealthy_test.csv datasets/jamming_features_research_moderate_test.csv datasets/jamming_features_research_severe_test.csv datasets/jamming_features_research_critical_test.csv datasets/jamming_features_research_ultra_stealthy_test.csv datasets/jamming_features_research_ultra_strong_test.csv --output results/evaluation_standard_threshold/seed_${seed}; done
```

- For evaluating model on stealthy-optimised threshold:

```
for seed in 42 123 456 789 999; do python3 evaluation.py --threshold-mode stealthy --model results/rq2/seed_${seed}/model_artifacts.joblib --bootstrap-iters 1000 --test-csvs datasets/jamming_features_research_stealthy_test.csv datasets/jamming_features_research_moderate_test.csv datasets/jamming_features_research_severe_test.csv datasets/jamming_features_research_critical_test.csv datasets/jamming_features_research_ultra_stealthy_test.csv datasets/jamming_features_research_ultra_strong_test.csv --output results/evaluation_stealthy_threshold/seed_${seed}; done
```

Where --model is the path to the stored model artifacts and output is the path to where evaluation results are stored (recommend to keep seperate)

aggregate_seeds.py

- Script to consolidate/aggregate/summarise results across the multiple seed runs, for organisation and some plots

```
python3 aggregate_seeds.py --training-root results/rq2 --standard-root results/evaluation_standard_threshold --stealthy-root results/evaluation_stealthy_threshold --outdir results/aggregate_results
```

src/ contains helper classes for my ML pipeline: data_handler.py, models.py, timing.py, metrics.py and utils.py

**Note** for the runs, output directory set to results/ different so it doesn't override my experimental results

### Other

- The `datasets` folder contain train (train_jamming_features_research.csv) and test datasets (\*\_test.csv)
- For transparency, you can find the folder of the full experimental results output used for analysis and reported in dissertation. This is located in `experimental_results` folder. Subfolder 'rq2' was for RQ2, 'eval' was for evaluation (RQ3) standard threhsold, 'eval_stealthy_threshold' was for evaluation (RQ3) stealthy threshold. and 'active_jamming' for comparison

## References

1. Lyu et al. "RIS-Based Wireless Jamming Attacks" IEEE Wireless Communications Letters 2020, IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1663-1667, Oct. 2020.

---

## Important

This is only for academic research use.
