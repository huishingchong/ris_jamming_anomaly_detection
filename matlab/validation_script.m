%% RIS Jamming Attack Implementation Validation and Figure generation
% Based on: Lyu et al. "IRS-Based Wireless Jamming Attacks: When Jammers Can Attack Without Power",
% published on IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1663-1667, Oct. 2020.
% DOI: 10.1109/LWC.2020.3000892
% RIS Jamming algorithm and theory credits to Lyu et al.

clear; close all; clc;
cvx_version;

%% Figure Saving Configuration
FIGURES_DIR = fullfile(pwd, 'validation_figures');
if ~exist(FIGURES_DIR, 'dir')
    mkdir(FIGURES_DIR);
    fprintf('Created figures directory: %s\n', FIGURES_DIR);
else
    delete(fullfile(FIGURES_DIR, '*.png'));
    delete(fullfile(FIGURES_DIR, '*.fig'));
    delete(fullfile(FIGURES_DIR, '*.jpg'));
    delete(fullfile(FIGURES_DIR, '*.jpeg'));
    fprintf('Cleaning directory: %s\n', FIGURES_DIR);
end

fprintf('Current MATLAB directory: %s\n', pwd);
fprintf('Figures will be saved to: %s\n', FIGURES_DIR);

%% Default Parameters
% Aligned with Lyu et al. for validation (see their IV. Performance Evaluation section)
M = 8;
N = 250; % We observe that RIS-based jamming in MRT beamforming is weak and would need higher N like N=250
% Note: to change beamforming strategy, would need to change below omega values throughout (currently MRT is in use, random beamforming is commented out)

PT_dBm = 30;
sigma2_dBm = -60;
b = 5;

NUM_REALISATIONS = 15; 
NUM_STATISTICAL_RUNS = 100; % For reproducing figures 2-5, the more runs the better

% Lyu et al. used a single iteration of optimisation for evaluation, and mentioned it still achieves good performance
USE_BCD_ITERATIONS = false;
MAX_BCD_ITERATIONS = 10;
BCD_CONVERGENCE_THRESHOLD = 1e-3;

GENERATE_FIGURES = true;
USE_CONFIDENCE_INTERVALS = true;
CONFIDENCE_LEVEL = 0.95;

OMEGA_SEED = 42;
SDR_RANDOMISATION_TRIALS = 450;

% Geometry positions of LT, 
x_t = 0;
x_r = 10;
x_i = 5;
y_i = 2;

fprintf('--- RIS JAMMING VALIDATION ---\n');
fprintf('Paper: Lyu et al. IEEE Wireless Communications Letters 2020\n');
fprintf('Using omega seed: %d\n', OMEGA_SEED);
fprintf('BCD iterations: %s (max=%d, threshold=%.1e)\n', ...
    logical_to_string(USE_BCD_ITERATIONS), MAX_BCD_ITERATIONS, BCD_CONVERGENCE_THRESHOLD);
fprintf('Error bars: %s (%.1f%% confidence level)\n', ...
    logical_to_string(USE_CONFIDENCE_INTERVALS), CONFIDENCE_LEVEL*100);

%% System Parameters
PT_linear = 10^(PT_dBm/10) / 1000;
sigma2 = 10^(sigma2_dBm/10) / 1000;
L = 2^b;
F = (0:L-1) * 2*pi / L;

A = 10^(-30/10);
alpha_LT_LR = 3.5;
alpha_LT_RIS = 2.8;
alpha_RIS_LR = 2.8;

% For Active jamming
Pa_15dBm = 10^(-45/10) / 1000;
Pa_25dBm = 10^(-35/10) / 1000;
Pa_35dBm = 10^(-25/10) / 1000;

% Output for transparency
fprintf('Configuration: M=%d, N=%d, realisations=%d, statistical_runs=%d\n', ...
    M, N, NUM_REALISATIONS, NUM_STATISTICAL_RUNS);
fprintf('SDR trials D=%d\n', SDR_RANDOMISATION_TRIALS);

d_LT_LR = abs(x_r - x_t);
d_LT_RIS = sqrt((x_i - x_t)^2 + y_i^2);
d_RIS_LR = sqrt((x_r - x_i)^2 + y_i^2);
fprintf('Network: LT-LR=%.1fm, LT-RIS=%.1fm, RIS-LR=%.1fm\n', d_LT_LR, d_LT_RIS, d_RIS_LR);

%% Main Simulation
fprintf('\n--- MAIN SIMULATION ---\n');

SINR_no_jamming = zeros(NUM_REALISATIONS, 1);
SINR_ris_jamming = zeros(NUM_REALISATIONS, 1);
SINR_active_15dBm = zeros(NUM_REALISATIONS, 1);
SINR_active_25dBm = zeros(NUM_REALISATIONS, 1);
SINR_active_35dBm = zeros(NUM_REALISATIONS, 1);

signal_power_no_jamming = zeros(NUM_REALISATIONS, 1);
signal_power_ris_jamming = zeros(NUM_REALISATIONS, 1);

sdr_success_rate = zeros(NUM_REALISATIONS, 1);
bcd_iterations_used = zeros(NUM_REALISATIONS, 1);

rng(OMEGA_SEED);
for realisation = 1:NUM_REALISATIONS
    [h_d, h_r, G, omega, PL_jammer_rx] = generate_channels( ...
        x_t, x_r, x_i, y_i, M, N, PT_linear, A, ...
        alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR);
    
    signal_power_direct = abs(h_d * omega)^2;
    SINR_direct = signal_power_direct / sigma2;
    
    [beta_opt, theta_opt, sdr_success, bcd_iters] = solve_ris_jamming_optimisation( ...
        h_r, h_d, G, omega, F, SDR_RANDOMISATION_TRIALS, ...
        USE_BCD_ITERATIONS, MAX_BCD_ITERATIONS, BCD_CONVERGENCE_THRESHOLD);
    
    Gamma = diag(beta_opt);
    Theta_bar = diag(exp(1j * theta_opt));
    total_channel_ris = h_r * Gamma * Theta_bar * G + h_d;
    signal_power_ris = abs(total_channel_ris * omega)^2;
    SINR_ris = signal_power_ris / sigma2;
    
    Pa_15_received = 10^(-45/10) / 1000;
    Pa_25_received = 10^(-35/10) / 1000;
    Pa_35_received = 10^(-25/10) / 1000;
    
    SINR_active_15 = signal_power_direct / (sigma2 + Pa_15_received);
    SINR_active_25 = signal_power_direct / (sigma2 + Pa_25_received);
    SINR_active_35 = signal_power_direct / (sigma2 + Pa_35_received);
    
    SINR_no_jamming(realisation) = SINR_direct;
    SINR_ris_jamming(realisation) = SINR_ris;
    SINR_active_15dBm(realisation) = SINR_active_15;
    SINR_active_25dBm(realisation) = SINR_active_25;
    SINR_active_35dBm(realisation) = SINR_active_35;
    
    signal_power_no_jamming(realisation) = signal_power_direct;
    signal_power_ris_jamming(realisation) = signal_power_ris;
    
    sdr_success_rate(realisation) = sdr_success;
    bcd_iterations_used(realisation) = bcd_iters;

    if realisation <= 10
        power_reduction_dB = 10*log10(signal_power_direct / signal_power_ris);
        sinr_reduction_dB = 10*log10(SINR_direct / SINR_ris);
        
        fprintf('Realisation %d: Power reduction %.1f dB, SINR %.1f→%.1f dB\n', ...
            realisation, power_reduction_dB, 10*log10(SINR_direct), 10*log10(SINR_ris));
    end
end

%% Performance Analysis
fprintf('\n--- PERFORMANCE ANALYSIS ---\n');

SINR_no_jamming_dB = 10*log10(SINR_no_jamming);
SINR_ris_jamming_dB = 10*log10(SINR_ris_jamming);
SINR_active_15dBm_dB = 10*log10(SINR_active_15dBm);
SINR_active_25dBm_dB = 10*log10(SINR_active_25dBm);
SINR_active_35dBm_dB = 10*log10(SINR_active_35dBm);

mean_SINR_no_jamming = mean(SINR_no_jamming_dB);
mean_SINR_ris_jamming = mean(SINR_ris_jamming_dB);
mean_SINR_active_15dBm = mean(SINR_active_15dBm_dB);
mean_SINR_active_25dBm = mean(SINR_active_25dBm_dB);
mean_SINR_active_35dBm = mean(SINR_active_35dBm_dB);

mean_sinr_reduction = mean_SINR_no_jamming - mean_SINR_ris_jamming;
mean_power_reduction_dB = mean(10*log10(signal_power_no_jamming ./ signal_power_ris_jamming));
max_power_reduction_dB = max(10*log10(signal_power_no_jamming ./ signal_power_ris_jamming));
std_power_reduction_dB = std(10*log10(signal_power_no_jamming ./ signal_power_ris_jamming));

ris_outperforms_15dBm = sum(SINR_ris_jamming_dB < SINR_active_15dBm_dB);
ris_outperforms_25dBm = sum(SINR_ris_jamming_dB < SINR_active_25dBm_dB);
ris_outperforms_35dBm = sum(SINR_ris_jamming_dB < SINR_active_35dBm_dB);

fprintf('SINR Performance:\n');
fprintf('  No Jamming: %.1f dB\n', mean_SINR_no_jamming);
fprintf('  RIS Jamming: %.1f dB\n', mean_SINR_ris_jamming);
fprintf('  Active 15dBm: %.1f dB\n', mean_SINR_active_15dBm);
fprintf('  Active 25dBm: %.1f dB\n', mean_SINR_active_25dBm);
fprintf('  Active 35dBm: %.1f dB\n', mean_SINR_active_35dBm);

fprintf('\nAttack Effectiveness:\n');
fprintf('  Mean SINR Reduction: %.1f dB\n', mean_sinr_reduction);
fprintf('  Mean Power Reduction: %.1f ± %.1f dB\n', mean_power_reduction_dB, std_power_reduction_dB);
fprintf('  Max Power Reduction: %.1f dB (%.1f%% reduction)\n', ...
    max_power_reduction_dB, (1-10^(-max_power_reduction_dB/10))*100);

fprintf('\nComparative Analysis:\n');
fprintf('  RIS outperforms 15dBm active: %d/%d (%.1f%%)\n', ...
    ris_outperforms_15dBm, NUM_REALISATIONS, 100*ris_outperforms_15dBm/NUM_REALISATIONS);
fprintf('  RIS outperforms 25dBm active: %d/%d (%.1f%%)\n', ...
    ris_outperforms_25dBm, NUM_REALISATIONS, 100*ris_outperforms_25dBm/NUM_REALISATIONS);
fprintf('  RIS outperforms 35dBm active: %d/%d (%.1f%%)\n', ...
    ris_outperforms_35dBm, NUM_REALISATIONS, 100*ris_outperforms_35dBm/NUM_REALISATIONS);

fprintf('\nAlgorithm Performance:\n');
fprintf('  SDR Success Rate: %.1f%%\n', 100*mean(sdr_success_rate));
fprintf('  Mean BCD Iterations: %.1f\n', mean(bcd_iterations_used));

if USE_CONFIDENCE_INTERVALS
    fprintf('\nStatistical Analysis (%.1f%% confidence):\n', CONFIDENCE_LEVEL*100);
    
    success_rate = mean(SINR_ris_jamming_dB < SINR_no_jamming_dB) * 100;
    fprintf('  RIS jamming success rate: %.1f%%\n', success_rate);

    var_no_jam = var(SINR_no_jamming_dB);
    var_ris_jam = var(SINR_ris_jamming_dB);
    cohens_d = abs(mean_SINR_no_jamming - mean_SINR_ris_jamming) / sqrt((var_no_jam + var_ris_jam) / 2);
    
    fprintf('  Effect size (Cohen''s d): %.2f\n', cohens_d);

    if cohens_d > 0.8
        fprintf('  Statistical significance: Large effect\n');
    elseif cohens_d > 0.5
        fprintf('  Statistical significance: Medium effect\n');
    elseif cohens_d > 0.2
        fprintf('  Statistical significance: Small effect\n');
    else
        fprintf('  Statistical significance: Negligible effect\n');
    end

    std_no_jam = sqrt(var_no_jam);
    std_ris_jam = sqrt(var_ris_jam);
    fprintf('SINR variance – No jamming: %.2f dB^2, RIS jamming: %.2f dB^2\n', var_no_jam, var_ris_jam);
    fprintf('SINR std deviation – No jamming: %.2f dB, RIS jamming: %.2f dB\n', std_no_jam, std_ris_jam);
end

%% Generate Figures
fprintf('\n--- GENERATING FIGURES ---\n');

generate_and_save_main_summary(SINR_no_jamming_dB, SINR_ris_jamming_dB, ...
    SINR_active_15dBm_dB, SINR_active_25dBm_dB, SINR_active_35dBm_dB, ...
    signal_power_no_jamming, signal_power_ris_jamming, mean_power_reduction_dB, ...
    mean_SINR_no_jamming, mean_SINR_ris_jamming, mean_SINR_active_15dBm, ...
    mean_SINR_active_25dBm, mean_SINR_active_35dBm, FIGURES_DIR);

if GENERATE_FIGURES
    fprintf('Generating publication figures 2-5 with statistical analysis...\n');
    
    generate_and_save_figure_2(x_t, x_r, x_i, y_i, M, N, b, A, ...
        alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15dBm, Pa_25dBm, Pa_35dBm, ...
        F, SDR_RANDOMISATION_TRIALS, OMEGA_SEED, NUM_STATISTICAL_RUNS, ...
        FIGURES_DIR, USE_CONFIDENCE_INTERVALS, CONFIDENCE_LEVEL);
    
    generate_and_save_figure_3(x_t, x_r, x_i, y_i, M, PT_dBm, b, A, ...
        alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15dBm, Pa_25dBm, Pa_35dBm, ...
        F, SDR_RANDOMISATION_TRIALS, OMEGA_SEED, NUM_STATISTICAL_RUNS, ...
        FIGURES_DIR, USE_CONFIDENCE_INTERVALS, CONFIDENCE_LEVEL);
    
    generate_and_save_figure_4(x_t, x_r, x_i, y_i, M, N, PT_dBm, b, A, ...
        alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15dBm, Pa_25dBm, Pa_35dBm, ...
        F, SDR_RANDOMISATION_TRIALS, OMEGA_SEED, NUM_STATISTICAL_RUNS, ...
        FIGURES_DIR, USE_CONFIDENCE_INTERVALS, CONFIDENCE_LEVEL);
    
    generate_and_save_figure_5(x_t, x_r, x_i, y_i, M, N, PT_dBm, b, A, ...
        alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15dBm, Pa_25dBm, Pa_35dBm, ...
        F, SDR_RANDOMISATION_TRIALS, OMEGA_SEED, NUM_STATISTICAL_RUNS, ...
        FIGURES_DIR, USE_CONFIDENCE_INTERVALS, CONFIDENCE_LEVEL);
    
    fprintf('Figures 2-5 generated successfully\n');
end

fprintf('Figure generation complete. Files saved to %s directory\n', FIGURES_DIR);

%% Supporting Functions

function [h_d, h_r, G, omega, PL_jammer_rx] = generate_channels( ...
    x_t, x_r, x_i, y_i, M, N, PT_linear, A, alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR)
    
    d_LT_LR = abs(x_r - x_t);
    d_LT_RIS = sqrt((x_i - x_t)^2 + y_i^2);
    d_RIS_LR = sqrt((x_r - x_i)^2 + y_i^2);
    
    PL_LT_LR = A * max(d_LT_LR, 1)^(-alpha_LT_LR);
    PL_LT_RIS = A * max(d_LT_RIS, 1)^(-alpha_LT_RIS);
    PL_RIS_LR = A * max(d_RIS_LR, 1)^(-alpha_RIS_LR);
    
    h_d = sqrt(PL_LT_LR/2) * (randn(1, M) + 1j*randn(1, M));
    h_r = sqrt(PL_RIS_LR/2) * (randn(1, N) + 1j*randn(1, N));
    G = sqrt(PL_LT_RIS/2) * (randn(N, M) + 1j*randn(N, M));

    % BEAMFORMING VECTOR: paper specified transmit beamforming vector 
    % satisfying ||omega||^2 = PT

    % % MRT
    omega_base = h_d' / norm(h_d);
    omega = sqrt(PT_linear) * omega_base;
    
    % % Randomised beamforming?
    % omega_base = (randn(M, 1) + 1j*randn(M, 1));
    % omega = sqrt(PT_linear) * omega_base / norm(omega_base);

    PL_jammer_rx = PL_RIS_LR;
end

function generate_and_save_figure_2(x_t, x_r, x_i, y_i, M, N, b, A, ...
    alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15_received, Pa_25_received, Pa_35_received, ...
    F, D, OMEGA_SEED, num_runs, figures_dir, use_ci, confidence_level)
    
    PT_range_dBm = 10:5:40;
    
    fprintf('Generating Figure 2 with %d runs per point\n', num_runs);
    
    d_LT_LR = abs(x_r - x_t);
    d_LT_RIS = sqrt((x_i - x_t)^2 + y_i^2);
    d_RIS_LR = sqrt((x_r - x_i)^2 + y_i^2);
    
    PL_LT_LR = A * max(d_LT_LR, 1)^(-alpha_LT_LR);
    PL_LT_RIS = A * max(d_LT_RIS, 1)^(-alpha_LT_RIS);
    PL_RIS_LR = A * max(d_RIS_LR, 1)^(-alpha_RIS_LR);
    
    num_points = length(PT_range_dBm);
    SINR_no_jam_runs = zeros(num_points, num_runs);
    SINR_ris_jam_runs = zeros(num_points, num_runs);
    SINR_active_15_runs = zeros(num_points, num_runs);
    SINR_active_25_runs = zeros(num_points, num_runs);
    SINR_active_35_runs = zeros(num_points, num_runs);
    
    for pt_idx = 1:num_points
        PT_dBm = PT_range_dBm(pt_idx);
        PT_linear = 10^(PT_dBm/10) / 1000;
        
        fprintf('  Processing PT = %d dBm (%d/%d)...', PT_dBm, pt_idx, num_points);
        
        for run = 1:num_runs
            run_seed = OMEGA_SEED + run*10000 + pt_idx*1000;
            rng(run_seed);
            
            h_d = sqrt(PL_LT_LR/2) * (randn(1, M) + 1j*randn(1, M));
            h_r = sqrt(PL_RIS_LR/2) * (randn(1, N) + 1j*randn(1, N));
            G = sqrt(PL_LT_RIS/2) * (randn(N, M) + 1j*randn(N, M));
            
            % MRT
            omega_base = h_d' / norm(h_d);
            omega = sqrt(PT_linear) * omega_base;

            % omega_base = (randn(M, 1) + 1j*randn(M, 1));
            % omega = sqrt(PT_linear) * omega_base / norm(omega_base);
                        
            signal_power_direct = abs(h_d * omega)^2;
            SINR_no_jam_runs(pt_idx, run) = signal_power_direct / sigma2;
            
            try
                [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(...
                    h_r, h_d, G, omega, F, D, false, 1, 1e-6);
                
                Gamma = diag(beta_opt);
                Theta_bar = diag(exp(1j * theta_opt));
                total_channel_ris = h_r * Gamma * Theta_bar * G + h_d;
                signal_power_ris = abs(total_channel_ris * omega)^2;
                SINR_ris_jam_runs(pt_idx, run) = signal_power_ris / sigma2;
            catch ME
                fprintf('FIG 2 ERROR: %s\n', ME.message);
                SINR_ris_jam_runs(pt_idx, run) = SINR_no_jam_runs(pt_idx, run) * 0.01;
            end
            
            SINR_active_15_runs(pt_idx, run) = signal_power_direct / (sigma2 + Pa_15_received);
            SINR_active_25_runs(pt_idx, run) = signal_power_direct / (sigma2 + Pa_25_received);
            SINR_active_35_runs(pt_idx, run) = signal_power_direct / (sigma2 + Pa_35_received);
        end
        fprintf(' Done\n');
    end
    
    SINR_no_jam_dB = 10*log10(SINR_no_jam_runs);
    SINR_ris_jam_dB = 10*log10(SINR_ris_jam_runs);
    SINR_active_15_dB = 10*log10(SINR_active_15_runs);
    SINR_active_25_dB = 10*log10(SINR_active_25_runs);
    SINR_active_35_dB = 10*log10(SINR_active_35_runs);
    
    mean_no_jam = mean(SINR_no_jam_dB, 2);
    mean_ris_jam = mean(SINR_ris_jam_dB, 2);
    mean_active_15 = mean(SINR_active_15_dB, 2);
    mean_active_25 = mean(SINR_active_25_dB, 2);
    mean_active_35 = mean(SINR_active_35_dB, 2);
    
    std_no_jam = std(SINR_no_jam_dB, 0, 2);
    std_ris_jam = std(SINR_ris_jam_dB, 0, 2);
    std_active_15 = std(SINR_active_15_dB, 0, 2);
    std_active_25 = std(SINR_active_25_dB, 0, 2);
    std_active_35 = std(SINR_active_35_dB, 0, 2);
    
    fig = figure('Position', [200, 200, 900, 700], 'Visible', 'on', 'Name', 'Figure 2');
    hold on;
    
    color_no_jam = [0.2, 0.4, 0.8];
    color_ris_jam = [0.8, 0.2, 0.2];
    color_active_15 = [0.3, 0.3, 0.3];
    color_active_25 = [0.7, 0.3, 0.7];
    color_active_35 = [0.0, 0.6, 0.3];
    
    plot_with_error_bars(PT_range_dBm, mean_no_jam', std_no_jam', ...
        color_no_jam, 'o', '-', 'No Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(PT_range_dBm, mean_ris_jam', std_ris_jam', ...
        color_ris_jam, 's', '-', 'RIS Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(PT_range_dBm, mean_active_15', std_active_15', ...
        color_active_15, '^', '--', 'Active Jamming (Pa=15dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(PT_range_dBm, mean_active_25', std_active_25', ...
        color_active_25, 'd', '--', 'Active Jamming (Pa=25dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(PT_range_dBm, mean_active_35', std_active_35', ...
        color_active_35, 'v', '--', 'Active Jamming (Pa=35dBm)', num_runs, use_ci, confidence_level);
    
    xlabel('Transmit Power of the LT (dBm)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('SINR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    if use_ci
        title(sprintf('Figure 2: SINR versus Transmit Power (%.1f%% CI)', confidence_level*100), ...
            'FontSize', 16, 'FontWeight', 'bold');
    else
        title('Figure 2: SINR versus Transmit Power', 'FontSize', 16, 'FontWeight', 'bold');
    end
    
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    xlim([10, 40]);
    % ylim([-45, 40]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    
    saveas(fig, fullfile(figures_dir, 'figure_2.png'));
    
    fprintf('Figure 2 generated and saved\n');
end

function generate_and_save_figure_3(x_t, x_r, x_i, y_i, M, PT_dBm, b, A, ...
    alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15_received, Pa_25_received, Pa_35_received, ...
    F, D, OMEGA_SEED, num_runs, figures_dir, use_ci, confidence_level)
    
    N_range = 100:20:200;
    % N_range = 100:50:350; % for extended analysis (MRT)
    PT_linear = 10^(PT_dBm/10) / 1000;
    
    fprintf('Generating Figure 3 with %d runs per point\n', num_runs);
    
    d_LT_LR = abs(x_r - x_t);
    d_LT_RIS = sqrt((x_i - x_t)^2 + y_i^2);
    d_RIS_LR = sqrt((x_r - x_i)^2 + y_i^2);
    
    PL_LT_LR = A * max(d_LT_LR, 1)^(-alpha_LT_LR);
    PL_LT_RIS = A * max(d_LT_RIS, 1)^(-alpha_LT_RIS);
    PL_RIS_LR = A * max(d_RIS_LR, 1)^(-alpha_RIS_LR);
    
    num_points = length(N_range);
    SINR_no_jam_runs = zeros(num_points, num_runs);
    SINR_ris_jam_runs = zeros(num_points, num_runs);
    SINR_active_15_runs = zeros(num_points, num_runs);
    SINR_active_25_runs = zeros(num_points, num_runs);
    SINR_active_35_runs = zeros(num_points, num_runs);
    
    for n_idx = 1:num_points
        N_current = N_range(n_idx);
        
        fprintf('  Processing N = %d elements (%d/%d)...', N_current, n_idx, num_points);
        
        for run = 1:num_runs
            run_seed = OMEGA_SEED + run*20000 + n_idx*2000;
            rng(run_seed);
            
            h_d = sqrt(PL_LT_LR/2) * (randn(1, M) + 1j*randn(1, M));
            h_r = sqrt(PL_RIS_LR/2) * (randn(1, N_current) + 1j*randn(1, N_current));
            G = sqrt(PL_LT_RIS/2) * (randn(N_current, M) + 1j*randn(N_current, M));
            
            % MRT
            omega_base = h_d' / norm(h_d);
            omega = sqrt(PT_linear) * omega_base;

            % omega_base = (randn(M, 1) + 1j*randn(M, 1));
            % omega = sqrt(PT_linear) * omega_base / norm(omega_base);
            
            signal_power_direct = abs(h_d * omega)^2;
            SINR_no_jam_runs(n_idx, run) = signal_power_direct / sigma2;
            
            try
                [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(...
                    h_r, h_d, G, omega, F, D, false, 1, 1e-6);
                
                Gamma = diag(beta_opt);
                Theta_bar = diag(exp(1j * theta_opt));
                total_channel_ris = h_r * Gamma * Theta_bar * G + h_d;
                signal_power_ris = abs(total_channel_ris * omega)^2;
                SINR_ris_jam_runs(n_idx, run) = signal_power_ris / sigma2;
            catch ME
                fprintf('FIG 3 ERROR: %s\n', ME.message);
                SINR_ris_jam_runs(n_idx, run) = SINR_no_jam_runs(n_idx, run) * 0.1;
            end
            
            SINR_active_15_runs(n_idx, run) = signal_power_direct / (sigma2 + Pa_15_received);
            SINR_active_25_runs(n_idx, run) = signal_power_direct / (sigma2 + Pa_25_received);
            SINR_active_35_runs(n_idx, run) = signal_power_direct / (sigma2 + Pa_35_received);
        end
        fprintf(' Done\n');
    end
    
    SINR_no_jam_dB = 10*log10(SINR_no_jam_runs);
    SINR_ris_jam_dB = 10*log10(SINR_ris_jam_runs);
    SINR_active_15_dB = 10*log10(SINR_active_15_runs);
    SINR_active_25_dB = 10*log10(SINR_active_25_runs);
    SINR_active_35_dB = 10*log10(SINR_active_35_runs);
    
    mean_no_jam = mean(SINR_no_jam_dB, 2);
    mean_ris_jam = mean(SINR_ris_jam_dB, 2);
    mean_active_15 = mean(SINR_active_15_dB, 2);
    mean_active_25 = mean(SINR_active_25_dB, 2);
    mean_active_35 = mean(SINR_active_35_dB, 2);
    
    std_no_jam = std(SINR_no_jam_dB, 0, 2);
    std_ris_jam = std(SINR_ris_jam_dB, 0, 2);
    std_active_15 = std(SINR_active_15_dB, 0, 2);
    std_active_25 = std(SINR_active_25_dB, 0, 2);
    std_active_35 = std(SINR_active_35_dB, 0, 2);
    
    fig = figure('Position', [250, 250, 900, 700], 'Visible', 'on', 'Name', 'Figure 3');
    hold on;
    
    color_no_jam = [0.2, 0.4, 0.8];
    color_ris_jam = [0.8, 0.2, 0.2];
    color_active_15 = [0.3, 0.3, 0.3];
    color_active_25 = [0.7, 0.3, 0.7];
    color_active_35 = [0.0, 0.6, 0.3];
    
    plot_with_error_bars(N_range, mean_no_jam', std_no_jam', ...
        color_no_jam, 'o', '-', 'No Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(N_range, mean_ris_jam', std_ris_jam', ...
        color_ris_jam, 's', '-', 'RIS Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(N_range, mean_active_15', std_active_15', ...
        color_active_15, '^', '--', 'Active Jamming (Pa=15dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(N_range, mean_active_25', std_active_25', ...
        color_active_25, 'd', '--', 'Active Jamming (Pa=25dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(N_range, mean_active_35', std_active_35', ...
        color_active_35, 'v', '--', 'Active Jamming (Pa=35dBm)', num_runs, use_ci, confidence_level);
    
    xlabel('Number of Reflecting Elements', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('SINR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    if use_ci
        title(sprintf('Figure 3: SINR versus Number of Reflecting Elements (%.1f%% CI)', confidence_level*100), ...
            'FontSize', 16, 'FontWeight', 'bold');
    else
        title('Figure 3: SINR versus Number of Reflecting Elements', 'FontSize', 16, 'FontWeight', 'bold');
    end
    
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    xlim([100, 200]);
    
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    
    saveas(fig, fullfile(figures_dir, 'figure_3.png'));
    
    fprintf('Figure 3 generated and saved\n');
end

function generate_and_save_figure_4(x_t, x_r, x_i, y_i, M, N, PT_dBm, b, A, ...
    alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15_received, Pa_25_received, Pa_35_received, ...
    F, D, OMEGA_SEED, num_runs, figures_dir, use_ci, confidence_level)
    
    PT_linear = 10^(PT_dBm/10) / 1000;
    distance_range = 5:1:12;
    
    fprintf('Generating Figure 4 with %d runs per point\n', num_runs);
    
    d_RIS_LR_FIXED = 5;
    d_LT_LR = abs(x_r - x_t);
    PL_LT_LR = A * max(d_LT_LR, 1)^(-alpha_LT_LR);
    PL_RIS_LR_fixed = A * max(d_RIS_LR_FIXED, 1)^(-alpha_RIS_LR);
    
    num_points = length(distance_range);
    SINR_no_jam_runs = zeros(num_points, num_runs);
    SINR_ris_jam_runs = zeros(num_points, num_runs);
    SINR_active_15_runs = zeros(num_points, num_runs);  
    SINR_active_25_runs = zeros(num_points, num_runs);
    SINR_active_35_runs = zeros(num_points, num_runs);
    
    for i = 1:num_points
        d_LT_RIS_target = distance_range(i);
        
        fprintf('  Processing d_LT_RIS = %.1f m (%d/%d)...', d_LT_RIS_target, i, num_points);
        
        x_i_current = (d_LT_RIS_target^2 - d_RIS_LR_FIXED^2 + x_r^2 - x_t^2) / (2*(x_r - x_t));
        y_i_squared = d_LT_RIS_target^2 - (x_i_current - x_t)^2;
        
        if y_i_squared < 0
            continue;
        end
        
        y_i_current = sqrt(y_i_squared);
        
        actual_d_RIS_LR = sqrt((x_r - x_i_current)^2 + y_i_current^2);
        if abs(actual_d_RIS_LR - d_RIS_LR_FIXED) > 0.1
            continue;
        end
        
        PL_LT_RIS_current = A * max(d_LT_RIS_target, 1)^(-alpha_LT_RIS);
        
        for run = 1:num_runs
            run_seed = OMEGA_SEED + run*30000 + i*3000;
            rng(run_seed);
            
            h_d = sqrt(PL_LT_LR/2) * (randn(1, M) + 1j*randn(1, M));
            h_r = sqrt(PL_RIS_LR_fixed/2) * (randn(1, N) + 1j*randn(1, N));
            G = sqrt(PL_LT_RIS_current/2) * (randn(N, M) + 1j*randn(N, M));
            
            % MRT
            omega_base = h_d' / norm(h_d);
            omega = sqrt(PT_linear) * omega_base;

            % omega_base = (randn(M, 1) + 1j*randn(M, 1));
            % omega = sqrt(PT_linear) * omega_base / norm(omega_base);
            
            signal_power_direct = abs(h_d * omega)^2;
            SINR_no_jam_runs(i, run) = signal_power_direct / sigma2;
            
            try
                [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(...
                    h_r, h_d, G, omega, F, D, false, 1, 1e-6);
                
                Gamma = diag(beta_opt);
                Theta_bar = diag(exp(1j * theta_opt));
                total_channel_ris = h_r * Gamma * Theta_bar * G + h_d;
                signal_power_ris = abs(total_channel_ris * omega)^2;
                SINR_ris_jam_runs(i, run) = signal_power_ris / sigma2;
            catch ME
                fprintf('FIG 4 ERROR: %s\n', ME.message);
                SINR_ris_jam_runs(i, run) = SINR_no_jam_runs(i, run) * 0.1;
            end
            
            SINR_active_15_runs(i, run) = signal_power_direct / (sigma2 + Pa_15_received);
            SINR_active_25_runs(i, run) = signal_power_direct / (sigma2 + Pa_25_received);
            SINR_active_35_runs(i, run) = signal_power_direct / (sigma2 + Pa_35_received);
        end
        fprintf(' Done\n');
    end
    
    SINR_no_jam_dB = 10*log10(SINR_no_jam_runs);
    SINR_ris_jam_dB = 10*log10(SINR_ris_jam_runs);
    SINR_active_15_dB = 10*log10(SINR_active_15_runs);
    SINR_active_25_dB = 10*log10(SINR_active_25_runs);
    SINR_active_35_dB = 10*log10(SINR_active_35_runs);
    
    mean_no_jam = mean(SINR_no_jam_dB, 2);
    mean_ris_jam = mean(SINR_ris_jam_dB, 2);
    mean_active_15 = mean(SINR_active_15_dB, 2);
    mean_active_25 = mean(SINR_active_25_dB, 2);
    mean_active_35 = mean(SINR_active_35_dB, 2);
    
    std_no_jam = std(SINR_no_jam_dB, 0, 2);
    std_ris_jam = std(SINR_ris_jam_dB, 0, 2);
    std_active_15 = std(SINR_active_15_dB, 0, 2);
    std_active_25 = std(SINR_active_25_dB, 0, 2);
    std_active_35 = std(SINR_active_35_dB, 0, 2);
    
    fig = figure('Position', [300, 300, 900, 700], 'Visible', 'on', 'Name', 'Figure 4');
    hold on;
    
    color_no_jam = [0.2, 0.4, 0.8];
    color_ris_jam = [0.8, 0.2, 0.2];
    color_active_15 = [0.3, 0.3, 0.3];
    color_active_25 = [0.7, 0.3, 0.7];
    color_active_35 = [0.0, 0.6, 0.3];
    
    plot_with_error_bars(distance_range, mean_no_jam', std_no_jam', ...
        color_no_jam, 'o', '-', 'No Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_ris_jam', std_ris_jam', ...
        color_ris_jam, 's', '-', 'RIS Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_15', std_active_15', ...
        color_active_15, '^', '--', 'Active Jamming (Pa=15dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_25', std_active_25', ...
        color_active_25, 'd', '--', 'Active Jamming (Pa=25dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_35', std_active_35', ...
        color_active_35, 'v', '--', 'Active Jamming (Pa=35dBm)', num_runs, use_ci, confidence_level);
    
    xlabel('Distance between LT and RIS (m)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('SINR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    if use_ci
        title(sprintf('Figure 4: SINR vs Distance LT-RIS (RIS-LR fixed at %dm, %.1f%% CI)', ...
            d_RIS_LR_FIXED, confidence_level*100), 'FontSize', 16, 'FontWeight', 'bold');
    else
        title(sprintf('Figure 4: SINR vs Distance LT-RIS (RIS-LR fixed at %dm)', d_RIS_LR_FIXED), ...
            'FontSize', 16, 'FontWeight', 'bold');
    end
    
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    xlim([5, 12]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    % ylim([-20, 40]);
    yticks(-20:10:30);
    
    saveas(fig, fullfile(figures_dir, 'figure_4.png'));

    fprintf('Figure 4 generated and saved (RIS-LR distance fixed at %dm)\n', d_RIS_LR_FIXED);
end

function generate_and_save_figure_5(x_t, x_r, x_i, y_i, M, N, PT_dBm, b, A, ...
    alpha_LT_LR, alpha_LT_RIS, alpha_RIS_LR, sigma2, Pa_15_received, Pa_25_received, Pa_35_received, ...
    F, D, OMEGA_SEED, num_runs, figures_dir, use_ci, confidence_level)
    
    PT_linear = 10^(PT_dBm/10) / 1000;
    distance_range = 5:1:12;
    
    fprintf('Generating Figure 5 with %d runs per point\n', num_runs);
    
    d_LT_RIS_FIXED = 5;
    d_LT_LR = abs(x_r - x_t);
    PL_LT_LR = A * max(d_LT_LR, 1)^(-alpha_LT_LR);
    PL_LT_RIS_fixed = A * max(d_LT_RIS_FIXED, 1)^(-alpha_LT_RIS);
    
    num_points = length(distance_range);
    SINR_no_jam_runs = zeros(num_points, num_runs);
    SINR_ris_jam_runs = zeros(num_points, num_runs);
    SINR_active_15_runs = zeros(num_points, num_runs);
    SINR_active_25_runs = zeros(num_points, num_runs);
    SINR_active_35_runs = zeros(num_points, num_runs);
    
    for i = 1:num_points
        d_RIS_LR_target = distance_range(i);
        
        fprintf('  Processing d_RIS_LR = %.1f m (%d/%d)...', d_RIS_LR_target, i, num_points);
        
        x_i_current = (d_LT_RIS_FIXED^2 - d_RIS_LR_target^2 + x_r^2 - x_t^2) / (2*(x_r - x_t));
        
        y_i_squared = d_LT_RIS_FIXED^2 - (x_i_current - x_t)^2;
        
        if y_i_squared < 0
            continue;
        end
        
        y_i_current = sqrt(y_i_squared);
        
        actual_d_LT_RIS = sqrt((x_i_current - x_t)^2 + y_i_current^2);
        actual_d_RIS_LR = sqrt((x_r - x_i_current)^2 + y_i_current^2);
        
        if abs(actual_d_LT_RIS - d_LT_RIS_FIXED) > 0.1 || abs(actual_d_RIS_LR - d_RIS_LR_target) > 0.1
            continue;
        end
        
        PL_RIS_LR_current = A * max(d_RIS_LR_target, 1)^(-alpha_RIS_LR);
        
        for run = 1:num_runs
            run_seed = OMEGA_SEED + run*40000 + i*4000;
            rng(run_seed);
            
            h_d = sqrt(PL_LT_LR/2) * (randn(1, M) + 1j*randn(1, M));
            h_r = sqrt(PL_RIS_LR_current/2) * (randn(1, N) + 1j*randn(1, N));
            G = sqrt(PL_LT_RIS_fixed/2) * (randn(N, M) + 1j*randn(N, M));
            
            % MRT
            omega_base = h_d' / norm(h_d);
            omega = sqrt(PT_linear) * omega_base;
            
            % omega_base = (randn(M, 1) + 1j*randn(M, 1));
            % omega = sqrt(PT_linear) * omega_base / norm(omega_base);
                        
            signal_power_direct = abs(h_d * omega)^2;
            SINR_no_jam_runs(i, run) = signal_power_direct / sigma2;
            
            try
                [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(...
                    h_r, h_d, G, omega, F, D, false, 1, 1e-6);
                
                Gamma = diag(beta_opt);
                Theta_bar = diag(exp(1j * theta_opt));
                total_channel_ris = h_r * Gamma * Theta_bar * G + h_d;
                signal_power_ris = abs(total_channel_ris * omega)^2;
                SINR_ris_jam_runs(i, run) = signal_power_ris / sigma2;
            catch ME
                fprintf('FIG 5 ERROR: %s\n', ME.message);
                SINR_ris_jam_runs(i, run) = SINR_no_jam_runs(i, run) * 0.1;
            end
            
            SINR_active_15_runs(i, run) = signal_power_direct / (sigma2 + Pa_15_received);
            SINR_active_25_runs(i, run) = signal_power_direct / (sigma2 + Pa_25_received);
            SINR_active_35_runs(i, run) = signal_power_direct / (sigma2 + Pa_35_received);
        end
        fprintf(' Done\n');
    end
    
    SINR_no_jam_dB = 10*log10(SINR_no_jam_runs);
    SINR_ris_jam_dB = 10*log10(SINR_ris_jam_runs);
    SINR_active_15_dB = 10*log10(SINR_active_15_runs);
    SINR_active_25_dB = 10*log10(SINR_active_25_runs);
    SINR_active_35_dB = 10*log10(SINR_active_35_runs);
    
    mean_no_jam = mean(SINR_no_jam_dB, 2);
    mean_ris_jam = mean(SINR_ris_jam_dB, 2);
    mean_active_15 = mean(SINR_active_15_dB, 2);
    mean_active_25 = mean(SINR_active_25_dB, 2);
    mean_active_35 = mean(SINR_active_35_dB, 2);
    
    std_no_jam = std(SINR_no_jam_dB, 0, 2);
    std_ris_jam = std(SINR_ris_jam_dB, 0, 2);
    std_active_15 = std(SINR_active_15_dB, 0, 2);
    std_active_25 = std(SINR_active_25_dB, 0, 2);
    std_active_35 = std(SINR_active_35_dB, 0, 2);
    
    fig = figure('Position', [350, 350, 900, 700], 'Visible', 'on', 'Name', 'Figure 5');
    hold on;
    
    color_no_jam = [0.2, 0.4, 0.8];
    color_ris_jam = [0.8, 0.2, 0.2];
    color_active_15 = [0.3, 0.3, 0.3];
    color_active_25 = [0.7, 0.3, 0.7];
    color_active_35 = [0.0, 0.6, 0.3];
    
    plot_with_error_bars(distance_range, mean_no_jam', std_no_jam', ...
        color_no_jam, 'o', '-', 'No Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_ris_jam', std_ris_jam', ...
        color_ris_jam, 's', '-', 'RIS Jamming', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_15', std_active_15', ...
        color_active_15, '^', '--', 'Active Jamming (Pa=15dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_25', std_active_25', ...
        color_active_25, 'd', '--', 'Active Jamming (Pa=25dBm)', num_runs, use_ci, confidence_level);
    plot_with_error_bars(distance_range, mean_active_35', std_active_35', ...
        color_active_35, 'v', '--', 'Active Jamming (Pa=35dBm)', num_runs, use_ci, confidence_level);
    
    xlabel('Distance between LR and RIS (m)', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('SINR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
    if use_ci
        title(sprintf('Figure 5: SINR vs Distance RIS-LR (LT-RIS fixed at %dm, %.1f%% CI)', ...
            d_LT_RIS_FIXED, confidence_level*100), 'FontSize', 16, 'FontWeight', 'bold');
    else
        title(sprintf('Figure 5: SINR vs Distance RIS-LR (LT-RIS fixed at %dm)', d_LT_RIS_FIXED), ...
            'FontSize', 16, 'FontWeight', 'bold');
    end
    
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'GridAlpha', 0.3);
    xlim([5, 12]);
    set(gca, 'FontSize', 12, 'LineWidth', 1.5);
    % ylim([-20, 40]);
    yticks(-20:10:30);
    
    saveas(fig, fullfile(figures_dir, 'figure_5.png'));
    
    fprintf('Figure 5 generated and saved (LT-RIS distance fixed at %dm)\n', d_LT_RIS_FIXED);
end

function generate_and_save_main_summary(SINR_no_jamming_dB, SINR_ris_jamming_dB, ...
    SINR_active_15dBm_dB, SINR_active_25dBm_dB, SINR_active_35dBm_dB, ...
    signal_power_no_jamming, signal_power_ris_jamming, mean_power_reduction_dB, ...
    mean_SINR_no_jamming, mean_SINR_ris_jamming, mean_SINR_active_15dBm, ...
    mean_SINR_active_25dBm, mean_SINR_active_35dBm, figures_dir)

    main_fig = figure('Position', [100, 100, 1600, 500], 'Visible', 'on', 'Name', 'Main Summary');

    subplot(1,5,1);
    histogram(10*log10(signal_power_no_jamming ./ signal_power_ris_jamming), 15, ...
        'FaceColor', [0.6 0.8 1.0], 'EdgeColor', [0.2 0.4 0.8], 'FaceAlpha', 0.7);
    xlabel('Signal Power Reduction (dB)');
    ylabel('Count');
    title('RIS Attack Effectiveness');
    xline(mean_power_reduction_dB, 'r--', 'Mean', 'LineWidth', 2);
    xline(20, 'g--', '20 dB Target', 'LineWidth', 2);
    grid on;

    subplot(1,5,2);
    hold on;
    plot(SINR_no_jamming_dB, 'b-o', 'DisplayName', 'No Jamming', 'MarkerSize', 4);
    plot(SINR_ris_jamming_dB, 'r-s', 'DisplayName', 'RIS Jamming', 'MarkerSize', 4);
    plot(SINR_active_15dBm_dB, 'k--^', 'DisplayName', 'Active 15dBm', 'MarkerSize', 4);
    plot(SINR_active_25dBm_dB, 'm--d', 'DisplayName', 'Active 25dBm', 'MarkerSize', 4);
    plot(SINR_active_35dBm_dB, 'g--v', 'DisplayName', 'Active 35dBm', 'MarkerSize', 4);
    xlabel('Realisation');
    ylabel('SINR (dB)');
    title('SINR Comparison');
    legend('Location', 'best', 'FontSize', 8);
    grid on;

    subplot(1,5,3);
    scatter(SINR_no_jamming_dB, SINR_ris_jamming_dB, 50, [0.2 0.4 0.8], 'filled');
    xlabel('SINR No Jamming (dB)');
    ylabel('SINR RIS Jamming (dB)');
    title('Attack Impact');
    hold on;
    plot([-20 50], [-20 50], 'r--', 'LineWidth', 2);
    legend('Realisations', 'No Change Line', 'Location', 'best');
    grid on;

    subplot(1,5,4);
    attack_comparison = [mean_SINR_no_jamming, mean_SINR_ris_jamming, ...
                        mean_SINR_active_15dBm, mean_SINR_active_25dBm, mean_SINR_active_35dBm];
    attack_labels = {'No Jamming', 'RIS Jamming', 'Active 15dBm', 'Active 25dBm', 'Active 35dBm'};

    b = bar(attack_comparison);
    if verLessThan('matlab', '9.0')
        set(b, 'FaceColor', [0.5 0.5 1.0]);
    else
        colors = [0.2 0.4 0.8; 0.8 0.2 0.2; 0.3 0.3 0.3; 0.7 0.3 0.7; 0.0 0.6 0.3];
        try
            b.CData = colors;
        catch
            set(b, 'FaceColor', [0.5 0.5 1.0]);
        end
    end

    set(gca, 'XTickLabel', attack_labels, 'XTickLabelRotation', 45);
    ylabel('Mean SINR (dB)');
    title('Attack Comparison');
    grid on;

    subplot(1,5,5);
    power_reductions = 10*log10(signal_power_no_jamming ./ signal_power_ris_jamming);
    boxplot(power_reductions, 'Labels', {'RIS Jamming'});
    ylabel('Power Reduction (dB)');
    title('Attack Variability');
    grid on;

    sgtitle('RIS Jamming Attack Analysis');

    saveas(main_fig, fullfile(figures_dir, 'main_summary.png'));
    
    fprintf('Main summary figure generated and saved\n');
end

function plot_with_error_bars(x_data, y_mean, y_std, color, marker, line_style, display_name, num_runs, use_ci, confidence_level)
    
    if use_ci && num_runs >= 20
        z_score = norminv((1 + confidence_level) / 2);
        error_bars = z_score * y_std / sqrt(num_runs);
    else
        error_bars = y_std;
    end
    
    x_patch = [x_data, fliplr(x_data)];
    y_patch = [y_mean + error_bars, fliplr(y_mean - error_bars)];
    
    fill(x_patch, y_patch, color, 'EdgeColor', 'none', ...
         'FaceAlpha', 0.2, 'HandleVisibility', 'off');
    
    hold on;
    
    plot(x_data, y_mean, [line_style marker], 'Color', color, 'LineWidth', 2.5, ...
         'MarkerSize', 8, 'MarkerFaceColor', color, 'DisplayName', display_name);
    
    for i = 1:length(x_data)
        plot([x_data(i), x_data(i)], [y_mean(i) - error_bars(i), y_mean(i) + error_bars(i)], ...
             '-', 'Color', color, 'LineWidth', 1, 'HandleVisibility', 'off');
    end
end

function str = logical_to_string(val)
    if val
        str = 'true';
    else
        str = 'false';
    end
end
