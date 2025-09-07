# ris_jamming_anomaly_detection

Source code for Dissertation 'Machine Learning Anomaly Detection for Illegitimate RIS-based Jamming Attacks'
Author: Hui Shing Chong
Date: September 2025
Submitted in partial fulfillment of the requirements for the degree in Computing(Security & Reliability) MSc of Imperial College London

## Quick Start

### Prerequisites

- Python 3.8+
- MATLAB R2025a (optional, for full functionality)
- 4GB+ free disk space

## Project Structure

### MATLAB

MATLAB (.mat files) is used during the implementation of Lyu et al's [1] RIS-based jamming algorithm and its validation. For our ML pipeline, MATLAB scripts are also used to generate raw signals for training and test (stored in .mat), then converting them to datasets (.csv) for ML detection.

### Python

Python scripts are then used for the remaining of the pipeline - using the csv datasets for training, evaluation and produce experimental results.

## Installation

1. **Clone or download the project**

   ```bash
   git clone <repository-url>
   cd ris_jamming_anomaly_detection
   ```

2. `pip3 install requirements.txt`

3. Run MATLAB scripts by opening directory on MATLAB software, usually calling the script on the command window. MATLAB beginner-friendly guide here: https://matlabacademy.mathworks.com/details/matlab-onramp/gettingstarted

4. Run python scripts in the current directory to run your training and evaluation of models. This is used for experimental results

For file explanations refer to the `File/Folder Explanations` section below for helpful descriptions on how to run each script and what to expect.

## Dependencies

### CVX - Convex Optimization Toolbox

To run the RIS-based jamming simulation (e.g. validation_script.m, generate_raw_signals_stratified.m) on MATLAB, CVX is needed, which is a tool used for convex optimisation, as part of the algorithm described by Lyu et al. [1]
The CVX module is included in this repository for your convenience. **To set up** cd to the cvx directory and run cvx_setup, ensure you see that a cvx mode is available.

If this approach doesn't work for your system, users can alternatively install CVX system-wide.

- **Location**: `matlab/external/cvx/`
- **Version**: 2.1
- **License**: GPL
- **Purpose**: Semi-definite relaxation optimization for RIS phase design
- **Citation**: Grant & Boyd, CVX: Matlab software for disciplined convex programming
- **Website**: http://cvxr.com/cvx/

To download: https://github.com/cvxr/CVX/releases

Note: For mac users trying to donwload this: CVX installation may require allowing unverified applications through System Preferences > Security \& Privacy due to MacOS protection policies: https://macpaw.com/how-to/fix-macos-cannot-verify-that-app-is-free-from-malware OS might prevent cvx from being downloaded

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
- Input: .mat file, Output: .csv file

extract_features_dataset.m

- Helper function to extract and compute features from signals, used by generate_features_dataset_research.m
- Not standalone script to run

Other helper standalone scripts: distribution_analysis.m, explore_mat_file.m which I used for checks

### Python

In src:

- data_handler.py
- models.py
-

### Other

- datasets
- For transparency, you can find the folder of the full experimental result run used and reported in dissertation. This is located at `experimental_results`

## References

1. Lyu et al. "RIS-Based Wireless Jamming Attacks" IEEE Wireless Communications Letters 2020

---

## Important

## This is only for academic research use.

**Note**: This platform is designed for academic research on RIS security. Ensure ethical use and compliance with applicable regulations.

All scripts and steps to reproduce the above to get this running is ran on Macbook. Tested on...
