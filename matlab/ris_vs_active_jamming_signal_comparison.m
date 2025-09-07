%% RIS vs ACTIVE JAMMING PRELIMINARY ANALYSIS
% Signal characteristics analysis for feature engineering
% Three-scenario comparison: Normal, RIS Jamming, Active Jamming
% RIS jamming samples generated based on Lyu et al. "IRS-Based Wireless Jamming Attacks: When Jammers Can Attack Without Power",
% that was published on IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1663-1667, Oct. 2020.
% DOI: 10.1109/LWC.2020.3000892

clear all; close all; clc;

fprintf('RIS vs ACTIVE JAMMING PRELIMINARY ANALYSIS\n');
fprintf('Signal characteristics analysis for feature engineering\n\n');

%% Configuration Params
CONFIG = struct();
CONFIG.SAMPLE_RATE = 1e6; % 1 MHz sampling
CONFIG.SYMBOL_RATE = 200e3; % 200 kHz symbol rate  
CONFIG.NUM_SYMBOLS = 1000; % 1000 symbols per signal
CONFIG.MODULATION = '16QAM';

CONFIG.M = 8; % Number of transmit antennas
CONFIG.N = 250; % Number of RIS elements
CONFIG.PT_dBm = 30; % Transmit power
CONFIG.sigma2_dBm = -60; % Noise power
CONFIG.b = 5; % Phase shift quantisation bits

% Network geometry ranges
CONFIG.X_RANGE = [9, 12];
CONFIG.RIS_X_RANGE = [0.3, 0.7];
CONFIG.RIS_Y_RANGE = [1, 3];
CONFIG.RANDOMISE_POSITIONS = true;

% Target effectiveness (for fair comparison)
ris_effectiveness_target = [20, 21];
active_effectiveness_target = [20, 21];

fprintf('Analysis Parameters:\n');
fprintf('  RIS target effectiveness: %.1f-%.1f dB SINR reduction\n', ris_effectiveness_target(1), ris_effectiveness_target(2));
fprintf('  Active target effectiveness: %.1f-%.1f dB \n', active_effectiveness_target(1), active_effectiveness_target(2));

%% System Params
derived_params = struct();
derived_params.PT_linear = 10^(CONFIG.PT_dBm/10) / 1000;
derived_params.sigma2 = 10^(CONFIG.sigma2_dBm/10) / 1000;
derived_params.L = 2^CONFIG.b;
derived_params.F = (0:derived_params.L-1) * 2*pi / derived_params.L;

% Path-loss params
derived_params.A = 10^(-30/10);
derived_params.alpha_LT_LR = 3.5;
derived_params.alpha_LT_RIS = 2.8;
derived_params.alpha_RIS_LR = 2.8;

% Signal params
derived_params.SAMPLES_PER_SYMBOL = CONFIG.SAMPLE_RATE / CONFIG.SYMBOL_RATE;
derived_params.SIGNAL_LENGTH = CONFIG.NUM_SYMBOLS * derived_params.SAMPLES_PER_SYMBOL;

% Generate pulse shaping filter
try
    pulse_shaping_filter = rcosdesign(0.35, 6, derived_params.SAMPLES_PER_SYMBOL);
catch
    pulse_shaping_filter = ones(derived_params.SAMPLES_PER_SYMBOL, 1) / sqrt(derived_params.SAMPLES_PER_SYMBOL);
    fprintf('Using fallback pulse shaping filter\n');
end

%% Generate Signals
ANALYSIS_SEED = 123;
rng(ANALYSIS_SEED);
fprintf('Using seed %d for reproducible analysis\n', ANALYSIS_SEED);

% Generate geometry and channels
if CONFIG.RANDOMISE_POSITIONS
    [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
else
    x_t = 0; x_r = 10; x_i = 5; y_i = 2;
end

fprintf('Geometry: LT(%.1f, 0), LR(%.1f, 0), RIS(%.1f, %.1f)\n', x_t, x_r, x_i, y_i);

[h_d, h_r, G, omega, PL_jammer_rx] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
    derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
    derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);

% Generate 16-QAM signal
tx_signal = generate_qam_signal(CONFIG, derived_params, pulse_shaping_filter);

%% Scenario 1: Normal communication
rx_signal_clean = tx_signal * (h_d * omega);
direct_link_signal_power = abs(h_d * omega)^2;
noise_power = derived_params.sigma2;

noise = sqrt(noise_power/2) * (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
rx_normal = rx_signal_clean + noise;

baseline_snr_db = 10*log10(direct_link_signal_power / noise_power);

fprintf('\nNormal Communication:\n');
fprintf('  SNR: %.1f dB\n', baseline_snr_db);
fprintf('  Direct-link signal power (pre-noise): %.1f dBm\n', 10*log10(direct_link_signal_power*1000));

fprintf('\nNote: Depending on hardware, might take a while to get matching ris and active jamming signals for fair comparison.\n');
%% Scenario 2: RIS Jamming
fprintf('\nGenerating RIS jamming (target: %.1f-%.1f dB effectiveness)...\n', ris_effectiveness_target(1), ris_effectiveness_target(2));

max_ris_attempts = 500;
ris_found = false;

for attempt = 1:max_ris_attempts
    % Try different geometries for target effectiveness
    if CONFIG.RANDOMISE_POSITIONS && attempt > 1
        [x_t_ris, x_r_ris, x_i_ris, y_i_ris] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
        [h_d_ris, h_r_ris, G_ris, omega_ris, ~] = generate_channels(x_t_ris, x_r_ris, x_i_ris, y_i_ris, ...
            CONFIG.M, CONFIG.N, derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
            derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);
    else
        h_d_ris = h_d; h_r_ris = h_r; G_ris = G; omega_ris = omega;
    end
    
    % RIS jamming optimisation
    [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(h_r_ris, h_d_ris, G_ris, omega_ris, ...
        derived_params.F, 1000, false, 20, 1e-3);
    
    % Calculate effectiveness
    signal_power_direct_ris = abs(h_d_ris * omega_ris)^2;
    Gamma = diag(beta_opt);
    Theta = diag(exp(1j * theta_opt));
    total_channel_ris = h_r_ris * Gamma * Theta * G_ris + h_d_ris;
    signal_power_ris = abs(total_channel_ris * omega_ris)^2;
    
    baseline_snr_ris = 10*log10(signal_power_direct_ris / noise_power);
    sinr_ris = 10*log10(signal_power_ris / noise_power);
    effectiveness_ris = baseline_snr_ris - sinr_ris;
    
    % Check if in target range
    if effectiveness_ris >= ris_effectiveness_target(1) && effectiveness_ris <= ris_effectiveness_target(2)
        ris_found = true;
        
        rx_signal_clean_ris = tx_signal * (total_channel_ris * omega_ris);
        noise_ris = sqrt(noise_power/2) * (randn(size(rx_signal_clean_ris)) + 1j*randn(size(rx_signal_clean_ris)));
        rx_ris = rx_signal_clean_ris + noise_ris;
        
        fprintf('  Found RIS effectiveness: %.1f dB (attempt %d)\n', effectiveness_ris, attempt);
        fprintf('  RIS SINR: %.1f dB\n', sinr_ris);
        break;
    end
    
    if mod(attempt, 50) == 0
        fprintf('  RIS attempts: %d (best effectiveness: %.1f dB)\n', attempt, effectiveness_ris);
    end
end

if ~ris_found
    warning('Could not find RIS effectiveness in target range, using best attempt (%.1f dB)', effectiveness_ris);
    rx_signal_clean_ris = tx_signal * (total_channel_ris * omega_ris);
    noise_ris = sqrt(noise_power/2) * (randn(size(rx_signal_clean_ris)) + 1j*randn(size(rx_signal_clean_ris)));
    rx_ris = rx_signal_clean_ris + noise_ris;
end

%% Scenario 3: Active Jamming
fprintf('\nGenerating active jamming (targeting: %.1f-%.1f dB effectiveness)...\n', active_effectiveness_target(1), active_effectiveness_target(2));

% Generate jammer channel
h_jammer = generate_jammer_channel(PL_jammer_rx);

% Search for target effectiveness within power constraints
pmin = 15;
pmax = 45;
max_active_attempts = 100;
active_found = false;

for attempt = 1:max_active_attempts
    % Maybe try different jammer channels if needed attempts
    if attempt > 1
        h_jammer = generate_jammer_channel(PL_jammer_rx);
    end
    
    pmin_temp = pmin;
    pmax_temp = pmax;
    
    for k = 1:20
        pdBm = (pmin_temp + pmax_temp) / 2;
        jamming_power_tx_linear = 10^(pdBm/10) / 1000;
        
        jamming_signal_baseband = sqrt(jamming_power_tx_linear) * ...
            (randn(size(tx_signal)) + 1j*randn(size(tx_signal))) / sqrt(2);
        jamming_signal_at_rx = jamming_signal_baseband * h_jammer;
        jamming_power_rx = mean(abs(jamming_signal_at_rx).^2);
        
        active_sinr_linear_temp = direct_link_signal_power / (noise_power + jamming_power_rx);
        effectiveness_temp = baseline_snr_db - 10*log10(active_sinr_linear_temp);
        
        if effectiveness_temp < active_effectiveness_target(1)
            pmin_temp = pdBm;
        else
            pmax_temp = pdBm;
        end
    end
    
    jamming_power_dbm = pdBm;
    jamming_power_tx_linear = 10^(jamming_power_dbm/10) / 1000;
    
    % Generate final signal and check if in target range
    jamming_signal_baseband = sqrt(jamming_power_tx_linear) * ...
        (randn(size(tx_signal)) + 1j*randn(size(tx_signal))) / sqrt(2);
    jamming_signal_at_rx = jamming_signal_baseband * h_jammer;
    jamming_power_rx = mean(abs(jamming_signal_at_rx).^2);
    
    active_sinr_linear = direct_link_signal_power / (noise_power + jamming_power_rx);
    active_sinr_db = 10*log10(active_sinr_linear);
    active_effectiveness = baseline_snr_db - active_sinr_db;
    
    if active_effectiveness >= active_effectiveness_target(1) && active_effectiveness <= active_effectiveness_target(2)
        active_found = true;
        fprintf('  Found active jamming effectiveness: %.1f dB (attempt %d)\n', active_effectiveness, attempt);
        break;
    end
end

% Active jamming received signal
noise_active = sqrt(noise_power/2) * (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
rx_active = rx_signal_clean + jamming_signal_at_rx + noise_active;

fprintf('  Targeted jamming power TX: %.1f dBm\n', jamming_power_dbm);
fprintf('  Jamming power RX: %.1f dBm\n', 10*log10(jamming_power_rx*1000));
fprintf('  Active SINR: %.1f dB\n', active_sinr_db);
fprintf('  Active effectiveness: %.1f dB (target: %.1f-%.1f dB)\n', active_effectiveness, active_effectiveness_target(1), active_effectiveness_target(2));

%% Signal Characteristic Analysis
fprintf('\n--- SIGNAL CHARACTERISTIC ANALYSIS ---\n');

% Calculate key metrics
total_received_power_normal = mean(abs(rx_normal).^2);
total_received_power_ris = mean(abs(rx_ris).^2);
total_received_power_active = mean(abs(rx_active).^2);

amp_normal = mean(abs(rx_normal));
amp_ris = mean(abs(rx_ris));
amp_active = mean(abs(rx_active));

% Calculate power reduction in dB (as done in validation_script.m)
power_reduction_ris_db = 10*log10(total_received_power_normal / total_received_power_ris);
power_increase_active_db = 10*log10(total_received_power_active / total_received_power_normal);

fprintf('Total Received Power (post-noise/jam):\n');
fprintf('  Normal: %.1f dBm\n', 10*log10(total_received_power_normal*1000));
fprintf('  RIS: %.1f dBm (%.1f dB reduction)\n', 10*log10(total_received_power_ris*1000), power_reduction_ris_db);
fprintf('  Active: %.1f dBm (+%.1f dB increase)\n', 10*log10(total_received_power_active*1000), power_increase_active_db);

fprintf('\nAmplitude Analysis (mean magnitude):\n');
fprintf('  Normal: %.6f\n', amp_normal);
fprintf('  RIS: %.6f (%.1f%% reduction from normal)\n', amp_ris, 100*(1 - amp_ris/amp_normal));
fprintf('  Active: %.6f (%.1f%% increase from normal)\n', amp_active, 100*(amp_active/amp_normal - 1));

% Calculate spectral characteristics for terminal output
window_length = min(256, floor(length(rx_normal)/8));
overlap = floor(window_length/2);
nfft = 512;

[psd_normal, f] = pwelch(rx_normal, window_length, overlap, nfft, CONFIG.SAMPLE_RATE);
[psd_ris, ~] = pwelch(rx_ris, window_length, overlap, nfft, CONFIG.SAMPLE_RATE);
[psd_active, ~] = pwelch(rx_active, window_length, overlap, nfft, CONFIG.SAMPLE_RATE);

% Calculate spectral flatness using geometric mean
geo_mean = @(x) exp(mean(log(x + eps)));
spectral_flatness_normal = geo_mean(psd_normal) / mean(psd_normal);
spectral_flatness_ris = geo_mean(psd_ris) / mean(psd_ris);
spectral_flatness_active = geo_mean(psd_active) / mean(psd_active);

fprintf('\nSpectral Analysis:\n');
fprintf('  Normal flatness: %.3f\n', spectral_flatness_normal);
fprintf('  RIS flatness: %.3f\n', spectral_flatness_ris);
fprintf('  Active flatness: %.3f\n', spectral_flatness_active);

%% Visualisation
fig = figure('Position', [100, 100, 1500, 900]);
t = tiledlayout(fig, 2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');

% Colours
colours = struct();
colours.normal = [0.3, 0.7, 0.3];
colours.ris = [0.85, 0.33, 0.1];
colours.active = [0.3, 0.45, 0.9];

%% (a) Time-domain amplitude
ax1 = nexttile(t, [1,2]); 
t_ms = (0:length(rx_normal)-1) / CONFIG.SAMPLE_RATE * 1000;
plot_length = min(1000, length(rx_normal)); % First 1ms
t_subset = t_ms(1:plot_length);

plot(t_subset, abs(rx_normal(1:plot_length)), 'Color', colours.normal, 'LineWidth', 2);
hold on;
plot(t_subset, abs(rx_ris(1:plot_length)), 'Color', colours.ris, 'LineWidth', 2);
plot(t_subset, abs(rx_active(1:plot_length)), 'Color', [0.3 0.45 0.9 0.45], 'LineWidth', 2);

title('(a) Time‑domain amplitude', 'FontWeight', 'bold', 'FontSize', 16);
xlabel('Time (ms)', 'FontSize', 13);
ylabel('Amplitude', 'FontSize', 13);
legend('Normal', 'RIS Jamming', 'Active Jamming', 'Location', 'best', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);

%% (b) Total received power
ax2 = nexttile(t, [1,2]); 
powers_db = [10*log10(total_received_power_normal*1000), 10*log10(total_received_power_ris*1000), 10*log10(total_received_power_active*1000)];

bar(powers_db, 'FaceColor', [0.65, 0.65, 0.85], 'EdgeColor', 'k', 'LineWidth', 1.2);
set(gca, 'XTickLabel', {'Normal', 'RIS Jamming', '  Active Jamming'});
title('(b) Total received power', 'FontWeight', 'bold', 'FontSize', 16);
ylabel('Power (dBm)', 'FontSize', 13);
grid on;


text(1, powers_db(1) + 1, sprintf('%.1f dBm', powers_db(1)), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12.2);
text(2, powers_db(2) + 1, sprintf('%.1f dBm\n(-%.1f dB)', powers_db(2), power_reduction_ris_db), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12.2);
text(3, powers_db(3) + 1, sprintf('%.1f dBm\n(+%.1f dB)', powers_db(3), power_increase_active_db), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 12.2);
set(gca, 'FontSize', 11);

%% (c) Power spectral density
ax3 = nexttile(t, [1,2]); 

f_khz = f / 1000;
plot(f_khz, 10*log10(psd_normal + eps), 'Color', colours.normal, 'LineWidth', 2.2);
hold on;
plot(f_khz, 10*log10(psd_ris + eps), 'Color', colours.ris, 'LineWidth', 2.2);
plot(f_khz, 10*log10(psd_active + eps), 'Color', colours.active, 'LineWidth', 2.2);

title('(c) Power spectral density', 'FontWeight', 'bold', 'FontSize', 16);
xlabel('Frequency (kHz)', 'FontSize', 13);
ylabel('PSD (dB/Hz)', 'FontSize', 13);
legend('Normal', 'RIS Jamming', 'Active Jamming', 'Location', 'best', 'FontSize', 12);
grid on;
xlim([0, CONFIG.SAMPLE_RATE/2000]);
set(gca, 'FontSize', 11);

%% (d) Amplitude distribution (with 95 percentile lines)
ax4 = nexttile(t, [1,4]); 

amp_normal_vals = abs(rx_normal);
amp_ris_vals = abs(rx_ris);
amp_active_vals = abs(rx_active);

max_amp = max([amp_normal_vals; amp_ris_vals; amp_active_vals]);
edges = linspace(0, max_amp, 40);

histogram(amp_normal_vals, edges, 'Normalization', 'probability', ...
    'FaceColor', colours.normal, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
histogram(amp_ris_vals, edges, 'Normalization', 'probability', ...
    'FaceColor', colours.ris, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
histogram(amp_active_vals, edges, 'Normalization', 'probability', ...
    'FaceColor', colours.active, 'FaceAlpha', 0.5, 'EdgeColor', 'none');

% Add te 95th percentile lines
p95_normal = prctile(amp_normal_vals, 95);
p95_ris = prctile(amp_ris_vals, 95);
p95_active = prctile(amp_active_vals, 95);

xline(p95_normal, '--', 'Color', colours.normal, 'LineWidth', 1.8);
xline(p95_ris, '--', 'Color', colours.ris, 'LineWidth', 1.8);
xline(p95_active, '--', 'Color', colours.active, 'LineWidth', 1.8);

title('(d) Amplitude distribution', 'FontWeight', 'bold', 'FontSize', 16);
xlabel('Amplitude', 'FontSize', 13);
ylabel('Probability', 'FontSize', 13);
legend('Normal', 'RIS Jamming', 'Active Jamming', 'Location', 'best', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 11);

%% (e) 16‑QAM constellation
ax5 = nexttile(t, [1,2]); 

% Downsample to symbol rate and normalise for visibility
downsample_factor = round(CONFIG.SAMPLE_RATE / CONFIG.SYMBOL_RATE);
constellation_normal = rx_normal(1:downsample_factor:end);
constellation_ris = rx_ris(1:downsample_factor:end);
constellation_active = rx_active(1:downsample_factor:end);

% Normalise all constellations to same power level for comparison
power_normal = mean(abs(constellation_normal).^2);
power_ris = mean(abs(constellation_ris).^2);
power_active = mean(abs(constellation_active).^2);

constellation_ris_norm = constellation_ris * sqrt(power_normal / power_ris);
constellation_active_norm = constellation_active * sqrt(power_normal / power_active);

n_symbols = min(150, length(constellation_normal));

scatter(real(constellation_normal(1:n_symbols)), imag(constellation_normal(1:n_symbols)), ...
    18, colours.normal, 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(real(constellation_ris_norm(1:n_symbols)), imag(constellation_ris_norm(1:n_symbols)), ...
    18, colours.ris, 'filled', 'MarkerFaceAlpha', 0.7);
scatter(real(constellation_active_norm(1:n_symbols)), imag(constellation_active_norm(1:n_symbols)), ...
    18, colours.active, 'filled', 'MarkerFaceAlpha', 0.7);

title('          (e) 16‑QAM constellation', 'FontWeight', 'bold', 'FontSize', 15);
xlabel('In‑Phase (I)', 'FontSize', 13);
ylabel('Quadrature (Q)', 'FontSize', 13);
legend('Normal', 'RIS Jamming', 'Active Jamming', 'Location', 'best', 'FontSize', 10);
grid on;
axis equal;
axis tight;
set(gca, 'FontSize', 11);

sgtitle(t, 'RIS vs Active Jamming: Signal Characteristics for Feature Engineering', ...
        'FontSize', 20, 'FontWeight', 'bold');

exportgraphics(fig, 'preliminary_analysis_ris_vs_active.png', 'Resolution', 300);

%% Analysis Summary
fprintf('\n--- Analysis Findings ---\n');

fprintf('\nPower-Based Discrimination:\n');
fprintf('  RIS: %.1f dB power reduction (total received power)\n', power_reduction_ris_db);
fprintf('  Active: +%.1f dB power increase (total received power)\n', power_increase_active_db);

fprintf('\nAmplitude-Based Discrimination:\n');
fprintf('  RIS: %.1f%% amplitude reduction (mean magnitude vs normal)\n', 100*(1 - amp_ris/amp_normal));
fprintf('  Active: %.1f%% amplitude increase (mean magnitude vs normal)\n', 100*(amp_active/amp_normal - 1));

fprintf('\nEffectiveness Comparison:\n');
fprintf('  RIS effectiveness: %.1f dB\n', effectiveness_ris);
fprintf('  Active effectiveness: %.1f dB\n', active_effectiveness);

fprintf('\nSpectral Characteristics:\n');
fprintf('  Normal flatness: %.3f\n', spectral_flatness_normal);
fprintf('  RIS flatness: %.3f\n', spectral_flatness_ris);
fprintf('  Active flatness: %.3f\n', spectral_flatness_active);

%% Local Functions

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

    % MRT beamforming
    omega_base = h_d' / norm(h_d);
    omega = sqrt(PT_linear) * omega_base;
    
    PL_jammer_rx = PL_RIS_LR;
end

function h_j = generate_jammer_channel(PL_jammer_rx)
    h_j = sqrt(PL_jammer_rx/2) * (randn + 1j*randn);
    
    % Add Rician fading
    rician_k_factor = 0.5 + 2.5 * rand();
    los_component = sqrt(rician_k_factor / (1 + rician_k_factor));
    nlos_component = sqrt(1 / (1 + rician_k_factor)) * (randn + 1j*randn) / sqrt(2);
    h_j = h_j * (los_component + nlos_component);
    
    % Add shadowing
    shadowing_std_db = 4 + 4 * rand();
    shadowing_factor = 10^(shadowing_std_db * randn / 20);
    h_j = h_j * sqrt(shadowing_factor);
end

function [x_t, x_r, x_i, y_i] = generate_random_geometry(x_range, ris_x_range, ris_y_range)
    max_attempts = 100;
    A = 10^(-30/10);
    alpha_LT_LR = 3.5;
    alpha_LT_RIS = 2.8;
    alpha_RIS_LR = 2.8;
    
    for attempt = 1:max_attempts
        lr_separation = x_range(1) + (x_range(2) - x_range(1)) * rand();
        x_t = 0;
        x_r = lr_separation;
        
        ris_x_fraction = ris_x_range(1) + (ris_x_range(2) - ris_x_range(1)) * rand();
        x_i = x_t + ris_x_fraction * (x_r - x_t);
        y_i = ris_y_range(1) + (ris_y_range(2) - ris_y_range(1)) * rand();
        
        d_lt_ris = sqrt((x_i - x_t)^2 + y_i^2);
        d_ris_lr = sqrt((x_r - x_i)^2 + y_i^2);
        d_lt_lr = abs(x_r - x_t);
        
        PL_LT_LR = A * max(d_lt_lr, 1)^(-alpha_LT_LR);
        PL_LT_RIS = A * max(d_lt_ris, 1)^(-alpha_LT_RIS);
        PL_RIS_LR = A * max(d_ris_lr, 1)^(-alpha_RIS_LR);
        
        pl_ratio_1 = 10*log10(PL_LT_RIS / PL_LT_LR);
        pl_ratio_2 = 10*log10(PL_RIS_LR / PL_LT_LR);
        
        if abs(pl_ratio_1) > 35 || abs(pl_ratio_2) > 35
            continue;
        end
        
        if (d_lt_ris + d_ris_lr) < d_lt_lr * 1.1
            continue;
        end
        
        return;
    end
    
    x_t = 0; x_r = 10; x_i = 5; y_i = 2;
    warning('Random geometry generation failed, using the default');
end

function tx_signal = generate_qam_signal(CONFIG, derived_params, pulse_filter)
    data_bits = randi([0, 1], CONFIG.NUM_SYMBOLS * 4, 1);
    tx_symbols = qammod(data_bits, 16, 'InputType', 'bit', 'UnitAveragePower', true);
    tx_signal_baseband = upfirdn(tx_symbols, pulse_filter, derived_params.SAMPLES_PER_SYMBOL);
    
    if length(tx_signal_baseband) > derived_params.SIGNAL_LENGTH
        tx_signal = tx_signal_baseband(1:derived_params.SIGNAL_LENGTH);
    elseif length(tx_signal_baseband) < derived_params.SIGNAL_LENGTH
        padding_length = derived_params.SIGNAL_LENGTH - length(tx_signal_baseband);
        tx_signal = [tx_signal_baseband; zeros(padding_length, 1)];
    else
        tx_signal = tx_signal_baseband;
    end
end
