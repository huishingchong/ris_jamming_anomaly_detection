%% MISO RIS JAMMING RAW SIGNAL GENERATOR
% Generate raw signals for RIS jamming detection research - stratified dataset
% RIS jamming samples generated based on Lyu et al. "IRS-Based Wireless Jamming Attacks: When Jammers Can Attack Without Power",
% published on IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1663-1667, Oct. 2020.
% DOI: 10.1109/LWC.2020.3000892

clear all; close all; clc;

%% Configurations
CONFIG = struct();

CONFIG.USE_STRATIFICATION = true;
CONFIG.INCLUDE_ACTIVE_JAMMING = true;
CONFIG.RANDOMISE_POSITIONS = true;

% Signal params
CONFIG.SAMPLE_RATE = 1e6; % 1 MHz sampling frequency
CONFIG.SYMBOL_RATE = 200e3; % 200 kHz symbol rate (200k symbols/sec)
CONFIG.NUM_SYMBOLS = 1000; % 1000 symbols per signal burst - 5ms signal duration
CONFIG.MODULATION = '16QAM'; % 4-bits per symbol

% Effectiveness bands
CONFIG.RIS_JAMMING_BANDS.stealthy = [3, 6]; 
CONFIG.RIS_JAMMING_BANDS.moderate = [6, 10];
CONFIG.RIS_JAMMING_BANDS.severe = [10, 15];
CONFIG.RIS_JAMMING_BANDS.critical = [15, 22];
CONFIG.BAND_NAMES = {'stealthy', 'moderate', 'severe', 'critical'};

CONFIG.ACTIVE_JAMMING_BANDS.stealthy = [3, 6];
CONFIG.ACTIVE_JAMMING_BANDS.moderate = [6, 10];
CONFIG.ACTIVE_JAMMING_BANDS.severe = [10, 15];
CONFIG.ACTIVE_JAMMING_BANDS.critical = [15, 22];
CONFIG.ACTIVE_BAND_NAMES = {'stealthy', 'moderate', 'severe', 'critical'};

CONFIG.JAMMER_POWER_DBM_MIN = 5;
CONFIG.JAMMER_POWER_DBM_MAX =  20;

% Network geometry ranges
CONFIG.X_RANGE = [9, 12];
CONFIG.RIS_X_RANGE = [0.3, 0.7];
CONFIG.RIS_Y_RANGE = [1, 3];

% System parameters (adapted from Lyu et al. Section IV)
CONFIG.M = 8;
CONFIG.N = 250;
CONFIG.PT_dBm = 30;
CONFIG.sigma2_dBm = -60;
CONFIG.b = 5;

% Optimisation settings (single bcd iteration for computational efficiency)
CONFIG.USE_BCD_ITERATIONS = false;
CONFIG.SDR_RANDOMISATION_TRIALS = 1000;
CONFIG.BCD_MAX_ITERATIONS = 20;
CONFIG.BCD_CONVERGENCE_THRESHOLD = 1e-3;

% Control parameters
CONFIG.MAX_ATTEMPTS_PER_BAND = 10000;

CONFIG.OUTPUT_DIR = 'signals';
CONFIG.RANDOM_SEED = 42;
% CONFIG.RANDOM_SEED = 123; % Seed used for seperate testing set

% Number of Samples configuration
if CONFIG.USE_STRATIFICATION
    CONFIG.NO_JAMMING_SAMPLES = 1000;
    CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND = 250; % Multiplied by number of bands (in our case, 4)
    CONFIG.ACTIVE_JAMMING_SAMPLES_PER_BAND = 250;
    CONFIG.TOTAL_RIS_TARGET = CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND * length(CONFIG.BAND_NAMES);
    CONFIG.TOTAL_ACTIVE_TARGET = CONFIG.ACTIVE_JAMMING_SAMPLES_PER_BAND * length(CONFIG.BAND_NAMES);
else
    CONFIG.NO_JAMMING_SAMPLES = 250;
    CONFIG.RIS_JAMMING_SAMPLES = 250;
    CONFIG.ACTIVE_JAMMING_SAMPLES = 250;
end

%% Initialisation
fprintf('--- RAW SIGNAL GENERATOR ---\n');
display_configuration(CONFIG);

if ~exist(CONFIG.OUTPUT_DIR, 'dir')
    mkdir(CONFIG.OUTPUT_DIR);
end
rng(CONFIG.RANDOM_SEED);

derived_params = initialise_derived_parameters(CONFIG);
[storage, tracking] = initialise_storage_and_tracking(CONFIG, derived_params);
precomputed = precompute_constants(CONFIG, derived_params);

%% Signal Generation
% No-jamming samples
fprintf('Generating %d no-jamming samples...\n', CONFIG.NO_JAMMING_SAMPLES);
storage = generate_no_jamming_signals(CONFIG, derived_params, precomputed, storage);

% RIS jamming samples
if CONFIG.USE_STRATIFICATION
    fprintf('\nGenerating stratified RIS jamming samples...\n');
    [storage, tracking] = generate_stratified_ris_signals(CONFIG, derived_params, precomputed, storage, tracking);
else
    fprintf('\nGenerating basic RIS jamming samples...\n');
    [storage, tracking] = generate_basic_ris_signals(CONFIG, derived_params, precomputed, storage, tracking);
end

% Active jamming samples
if CONFIG.INCLUDE_ACTIVE_JAMMING
    if CONFIG.USE_STRATIFICATION
        fprintf('\nGenerating stratified active jamming samples...\n');
        [storage, tracking] = generate_stratified_active_signals(CONFIG, derived_params, precomputed, storage, tracking);
    else
        fprintf('\nGenerating basic active jamming samples...\n');
        [storage, tracking] = generate_basic_active_signals(CONFIG, derived_params, precomputed, storage, tracking);
    end
end

%% Save Dataset
fprintf('\n--- SAVING DATASET ---\n');

[raw_signals, raw_metadata] = finalise_dataset(storage);

raw_dataset = struct();
raw_dataset.signals = raw_signals;
raw_dataset.metadata = raw_metadata;
raw_dataset.config = CONFIG;
raw_dataset.derived_params = derived_params;
raw_dataset.tracking = tracking;
raw_dataset.feature_extraction_params = storage.feature_extraction_params;
raw_dataset.generation_timestamp = datetime('now');

if CONFIG.USE_STRATIFICATION && CONFIG.INCLUDE_ACTIVE_JAMMING
    raw_dataset.stratification_type = 'stratified_with_active';
elseif CONFIG.USE_STRATIFICATION
    raw_dataset.stratification_type = 'stratified';
else
    raw_dataset.stratification_type = 'basic';
end

raw_signals_filename = fullfile(CONFIG.OUTPUT_DIR, generate_file_name(CONFIG));
save(raw_signals_filename, 'raw_dataset', '-v7.3');

display_generation_summary(CONFIG, raw_dataset, raw_signals_filename);

%% Local Functions
% Display for transparency
function display_configuration(CONFIG)
    fprintf('\nConfiguration:\n');
    fprintf('  Stratification: %s\n', yesno(CONFIG.USE_STRATIFICATION));
    fprintf('  Active jamming: %s\n', yesno(CONFIG.INCLUDE_ACTIVE_JAMMING));
    fprintf('  Randomised geometry: %s\n', yesno(CONFIG.RANDOMISE_POSITIONS));
    fprintf('  Modulation: %s\n', CONFIG.MODULATION);
    fprintf('  Signal: %d symbols, %.0f kHz rate, %.1f MHz sampling\n', ...
        CONFIG.NUM_SYMBOLS, CONFIG.SYMBOL_RATE/1e3, CONFIG.SAMPLE_RATE/1e6);
    
    if CONFIG.USE_STRATIFICATION
        fprintf('  Samples per band: %d RIS', CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND);
        if CONFIG.INCLUDE_ACTIVE_JAMMING
            fprintf(', %d active', CONFIG.ACTIVE_JAMMING_SAMPLES_PER_BAND);
        end
        fprintf('\n');
    end
end

function name = generate_file_name(CONFIG)
    base_name = 'raw_signals';
    
    if CONFIG.USE_STRATIFICATION
        base_name = [base_name, '_stratified'];
    else
        base_name = [base_name, '_natural'];
    end
    
    if isfield(CONFIG, 'RANDOM_SEED')
        base_name = [base_name, '_seed', num2str(CONFIG.RANDOM_SEED)];
    end
    
    name = [base_name, '.mat'];
end

function answer = yesno(logical_value)
    if logical_value
        answer = 'enabled';
    else
        answer = 'disabled';
    end
end

function derived_params = initialise_derived_parameters(CONFIG)
    derived_params = struct();
    
    derived_params.PT_linear = 10^(CONFIG.PT_dBm/10) / 1000;
    derived_params.sigma2 = 10^(CONFIG.sigma2_dBm/10) / 1000;
    derived_params.L = 2^CONFIG.b;
    derived_params.F = (0:derived_params.L-1) * 2*pi / derived_params.L;
    
    % Path-loss parameters
    derived_params.A = 10^(-30/10);
    derived_params.alpha_LT_LR = 3.5;
    derived_params.alpha_LT_RIS = 2.8;
    derived_params.alpha_RIS_LR = 2.8;
    
    % Modulation parameters
    switch CONFIG.MODULATION
        case '16QAM'
            derived_params.BITS_PER_SYMBOL = 4;
            derived_params.MODULATION_ORDER = 16;
        case '64QAM'
            derived_params.BITS_PER_SYMBOL = 6;
            derived_params.MODULATION_ORDER = 64;
        otherwise
            error('Unsupported modulation: %s', CONFIG.MODULATION);
    end
    
    % Signal parameters
    derived_params.SAMPLES_PER_SYMBOL = CONFIG.SAMPLE_RATE / CONFIG.SYMBOL_RATE;
    derived_params.SIGNAL_LENGTH = CONFIG.NUM_SYMBOLS * derived_params.SAMPLES_PER_SYMBOL;
    
    % Feature extraction parameters
    derived_params.FEATURE_FFT_SIZE = 4096; % FFT length: 4096
    derived_params.FEATURE_MAX_LAG = 1000; % Max correlation lag: 1000 samples (0.1 symbol)
    
    % Calculate total samples
    if CONFIG.USE_STRATIFICATION
        derived_params.total_target_samples = CONFIG.NO_JAMMING_SAMPLES + CONFIG.TOTAL_RIS_TARGET;
        if CONFIG.INCLUDE_ACTIVE_JAMMING
            derived_params.total_target_samples = derived_params.total_target_samples + CONFIG.TOTAL_ACTIVE_TARGET;
        end
    else
        derived_params.total_target_samples = CONFIG.NO_JAMMING_SAMPLES + CONFIG.RIS_JAMMING_SAMPLES;
        if CONFIG.INCLUDE_ACTIVE_JAMMING
            derived_params.total_target_samples = derived_params.total_target_samples + CONFIG.ACTIVE_JAMMING_SAMPLES;
        end
    end
    
    fprintf('Target samples: %d total\n', derived_params.total_target_samples);
end

function [storage, tracking] = initialise_storage_and_tracking(CONFIG, derived_params)
    storage = struct();
    storage.raw_signals = cell(derived_params.total_target_samples, 1);
    storage.sample_idx = 0;
    
    % Unified metadata template for consistent SINR-based analysis
    sample_metadata = struct();
    sample_metadata.sample_idx = 0;
    sample_metadata.scenario = '';
    sample_metadata.signal_power = 0;
    sample_metadata.interference_power = 0;
    sample_metadata.noise_power = 0;
    sample_metadata.sinr_db = 0;
    sample_metadata.baseline_snr_db = 0;
    sample_metadata.effectiveness_db = 0;
    sample_metadata.attack_band = '';
    sample_metadata.label = 0;
    sample_metadata.geometry = struct('x_t', 0, 'x_r', 0, 'x_i', 0, 'y_i', 0);
    
    storage.metadata = repmat(sample_metadata, derived_params.total_target_samples, 1);
    
    % Feature extraction parameters
    storage.feature_extraction_params = struct();
    storage.feature_extraction_params.fs = CONFIG.SAMPLE_RATE;
    storage.feature_extraction_params.fft_size = derived_params.FEATURE_FFT_SIZE;
    storage.feature_extraction_params.max_lag = derived_params.FEATURE_MAX_LAG;
    
    % Tracking statistics
    tracking = struct();
    tracking.debug_stats = struct();
    tracking.debug_stats.total_ris_samples = 0;
    tracking.debug_stats.total_active_samples = 0;
    tracking.debug_stats.ris_effectiveness = [];
    tracking.debug_stats.active_effectiveness = [];
    
    if CONFIG.USE_STRATIFICATION
        tracking.band_statistics = struct();
        for i = 1:length(CONFIG.BAND_NAMES)
            band_name = CONFIG.BAND_NAMES{i};
            tracking.band_statistics.(band_name) = struct();
            tracking.band_statistics.(band_name).target_count = CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND;
            tracking.band_statistics.(band_name).accepted_count = 0;
            tracking.band_statistics.(band_name).total_attempts = 0;
            tracking.band_statistics.(band_name).effectiveness_values = [];
        end
        
        if CONFIG.INCLUDE_ACTIVE_JAMMING
            tracking.active_band_statistics = struct();
            for i = 1:length(CONFIG.ACTIVE_BAND_NAMES)
                band_name = CONFIG.ACTIVE_BAND_NAMES{i};
                tracking.active_band_statistics.(band_name) = struct();
                tracking.active_band_statistics.(band_name).target_count = CONFIG.ACTIVE_JAMMING_SAMPLES_PER_BAND;
                tracking.active_band_statistics.(band_name).accepted_count = 0;
                tracking.active_band_statistics.(band_name).total_attempts = 0;
                tracking.active_band_statistics.(band_name).effectiveness_values = [];
            end
        end
    end
    
    fprintf('Allocated storage: %d signals\n', derived_params.total_target_samples);
end

function precomputed = precompute_constants(CONFIG, derived_params)    
    precomputed = struct();
    
    % Root raised cosine (RRC) pulse shaping for realistic baseband signals
    PULSE_SHAPE_SPAN = 6; % Filter span: 6 symbol periods
    PULSE_SHAPE_ROLLOFF = 0.35;
    try
        precomputed.RRC_FILTER = rcosdesign(PULSE_SHAPE_ROLLOFF, PULSE_SHAPE_SPAN, derived_params.SAMPLES_PER_SYMBOL);
    catch
        precomputed.RRC_FILTER = ones(derived_params.SAMPLES_PER_SYMBOL, 1) / sqrt(derived_params.SAMPLES_PER_SYMBOL);
        fprintf('  RRC filter: fallback implementation\n');
    end
end

function storage = generate_no_jamming_signals(CONFIG, derived_params, precomputed, storage)
    for i = 1:CONFIG.NO_JAMMING_SAMPLES
        if mod(i, max(1, floor(CONFIG.NO_JAMMING_SAMPLES/10))) == 0
            fprintf('  Progress: %d/%d\n', i, CONFIG.NO_JAMMING_SAMPLES);
        end
        
        % Generate geometry
        if CONFIG.RANDOMISE_POSITIONS
            [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
        else
            x_t = 0; x_r = 10; x_i = 5; y_i = 2;
        end
        
        current_geometry = struct('x_t', x_t, 'x_r', x_r, 'x_i', x_i, 'y_i', y_i);

        % Generate channels
        [h_d, ~, ~, omega, ~] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
            derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
            derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);
        
        tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed);

        % No jamming scenario
        rx_signal_clean = tx_signal * (h_d * omega);
        
        % Add AWGN noise
        noise_power = derived_params.sigma2;
        noise = sqrt(noise_power/2) * (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
        rx_signal = rx_signal_clean + noise;
        
        % Calculate baseline and actual SINR
        signal_power_direct = abs(h_d * omega)^2;
        actual_signal_power = mean(abs(rx_signal_clean).^2);
        baseline_snr_db = 10*log10(signal_power_direct / noise_power);
        sinr_db = baseline_snr_db;
        effectiveness_db = 0;

        if mod(i, max(1, floor(CONFIG.NO_JAMMING_SAMPLES/5))) == 0 || i <= 3
            fprintf('    No-jamming sample %d: LT(%.2f, 0), LR(%.2f, 0), Baseline: %.1fdB\n', i, x_t, x_r, baseline_snr_db);
        end
        
        % Store signal and metadata
        storage.sample_idx = storage.sample_idx + 1;
        storage.raw_signals{storage.sample_idx} = rx_signal;
        
        storage = write_metadata(storage, storage.sample_idx, 'no_jamming', actual_signal_power, 0, ...
            noise_power, sinr_db, baseline_snr_db, effectiveness_db, current_geometry, 'none', 0);
    end
end

function [storage, tracking] = generate_stratified_ris_signals(CONFIG, derived_params, precomputed, storage, tracking)
    for band_idx = 1:length(CONFIG.BAND_NAMES)
        band_name = CONFIG.BAND_NAMES{band_idx};
        target_range = CONFIG.RIS_JAMMING_BANDS.(band_name);
        
        fprintf('-Generating %s RIS jamming samples (%.1f-%.1f dB)...\n', band_name, target_range(1), target_range(2));
        
        attempts = 0;
        
        while tracking.band_statistics.(band_name).accepted_count < CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND
            attempts = attempts + 1;
            tracking.band_statistics.(band_name).total_attempts = attempts;
            
            if attempts > CONFIG.MAX_ATTEMPTS_PER_BAND
                warning('%s band: gave up after %d attempts', band_name, attempts);
                break;
            end
            
            if CONFIG.RANDOMISE_POSITIONS
                [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
            else
                x_t = 0; x_r = 10; x_i = 5; y_i = 2;
            end
            
            current_geometry = struct('x_t', x_t, 'x_r', x_r, 'x_i', x_i, 'y_i', y_i);
            
            [h_d, h_r, G, omega, ~] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
                derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
                derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);
            
            % Generate signal
            tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed);

            tracking.debug_stats.total_ris_samples = tracking.debug_stats.total_ris_samples + 1;
            
            % Optimise RIS configuration
            [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(h_r, h_d, G, omega, ...
                derived_params.F, CONFIG.SDR_RANDOMISATION_TRIALS, CONFIG.USE_BCD_ITERATIONS, ...
                CONFIG.BCD_MAX_ITERATIONS, CONFIG.BCD_CONVERGENCE_THRESHOLD);
            
            % Calculate unified effectiveness metrics before band testing
            noise_power = derived_params.sigma2;
            signal_power_direct = abs(h_d * omega)^2;
            Gamma = diag(beta_opt);
            Theta = diag(exp(1j * theta_opt));
            total_channel_ris = h_r * Gamma * Theta * G + h_d;
            signal_power_ris = abs(total_channel_ris * omega)^2;
            
            baseline_snr_db = 10*log10(signal_power_direct / noise_power);
            sinr_db = 10*log10(signal_power_ris / noise_power);
            effectiveness_db = baseline_snr_db - sinr_db;
            
            % Check if in target band
            attack_in_band = (effectiveness_db >= target_range(1)) && ...
                            (isinf(target_range(2)) || effectiveness_db < target_range(2));
            
            if attack_in_band
                tracking.band_statistics.(band_name).accepted_count = tracking.band_statistics.(band_name).accepted_count + 1;
                tracking.band_statistics.(band_name).effectiveness_values = [tracking.band_statistics.(band_name).effectiveness_values, effectiveness_db];
                
                if mod(tracking.band_statistics.(band_name).accepted_count, 50) == 0 || tracking.band_statistics.(band_name).accepted_count <= 3
                    fprintf('    RIS sample %d (%s): LT(%.2f, 0), LR(%.2f, 0), RIS(%.2f, %.2f), Reduction: %.1f dB\n', ...
                        tracking.band_statistics.(band_name).accepted_count, band_name, x_t, x_r, x_i, y_i, effectiveness_db);
                end

                % Generate received signal
                rx_signal_clean = tx_signal * (total_channel_ris * omega);
                actual_signal_power_ris = mean(abs(rx_signal_clean).^2);
                
                noise = sqrt(noise_power/2) * (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
                rx_signal = rx_signal_clean + noise;
                
                % Store signal and metadata
                storage.sample_idx = storage.sample_idx + 1;
                storage.raw_signals{storage.sample_idx} = rx_signal;
                
                storage = write_metadata(storage, storage.sample_idx, 'ris_jamming', actual_signal_power_ris, 0, ...
                    noise_power, sinr_db, baseline_snr_db, effectiveness_db, current_geometry, band_name, 1);
                
                tracking.debug_stats.ris_effectiveness = [tracking.debug_stats.ris_effectiveness, effectiveness_db];
                
                if mod(tracking.band_statistics.(band_name).accepted_count, 50) == 0
                    acceptance_rate = 100 * tracking.band_statistics.(band_name).accepted_count / attempts;
                    fprintf('  %s: %d/%d accepted (%.2f%% rate)\n', band_name, ...
                        tracking.band_statistics.(band_name).accepted_count, ...
                        CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND, acceptance_rate);
                end
            end
            
            if mod(attempts, 500) == 0
                current_rate = 100 * tracking.band_statistics.(band_name).accepted_count / attempts;
                fprintf('  %s: %d/%d after %d attempts (%.2f%% rate)\n', band_name, ...
                    tracking.band_statistics.(band_name).accepted_count, ...
                    CONFIG.SAMPLES_PER_EFFECTIVENESS_BAND, attempts, current_rate);
            end
        end
        
        fprintf('  %s completed: %d samples\n', band_name, tracking.band_statistics.(band_name).accepted_count);
    end
end

function [storage, tracking] = generate_basic_ris_signals(CONFIG, derived_params, precomputed, storage, tracking)
    for i = 1:CONFIG.RIS_JAMMING_SAMPLES
        if mod(i, max(1, floor(CONFIG.RIS_JAMMING_SAMPLES/10))) == 0
            fprintf('  RIS jamming progress: %d/%d\n', i, CONFIG.RIS_JAMMING_SAMPLES);
        end
        
        if CONFIG.RANDOMISE_POSITIONS
            [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
        else
            x_t = 0; x_r = 10; x_i = 5; y_i = 2;
        end
        
        current_geometry = struct('x_t', x_t, 'x_r', x_r, 'x_i', x_i, 'y_i', y_i);
        
        [h_d, h_r, G, omega, ~] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
            derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
            derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);

        tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed);
        
        tracking.debug_stats.total_ris_samples = tracking.debug_stats.total_ris_samples + 1;
        
        [beta_opt, theta_opt, ~, ~] = solve_ris_jamming_optimisation(h_r, h_d, G, omega, ...
            derived_params.F, CONFIG.SDR_RANDOMISATION_TRIALS, CONFIG.USE_BCD_ITERATIONS, ...
            CONFIG.BCD_MAX_ITERATIONS, CONFIG.BCD_CONVERGENCE_THRESHOLD);
        
        % Calculate unified effectiveness metrics
        noise_power = derived_params.sigma2;
        signal_power_direct = abs(h_d * omega)^2;
        Gamma = diag(beta_opt);
        Theta = diag(exp(1j * theta_opt));
        total_channel_ris = h_r * Gamma * Theta * G + h_d;
        
        rx_signal_clean = tx_signal * (total_channel_ris * omega);
        actual_signal_power_ris = mean(abs(rx_signal_clean).^2);
        
        baseline_snr_db = 10*log10(signal_power_direct / noise_power);
        sinr_db = 10*log10(actual_signal_power_ris / noise_power);
        effectiveness_db = baseline_snr_db - sinr_db;
        
        noise = sqrt(noise_power/2) * (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
        rx_signal = rx_signal_clean + noise;

        if mod(i, max(1, floor(CONFIG.RIS_JAMMING_SAMPLES/20))) == 0 || i <= 3
            fprintf('    RIS jamming sample %d: LT(%.2f, 0), LR(%.2f, 0), RIS(%.2f, %.2f), Reduction: %.1fdB\n', i, x_t, x_r, x_i, y_i, effectiveness_db);
        end
        
        storage.sample_idx = storage.sample_idx + 1;
        storage.raw_signals{storage.sample_idx} = rx_signal;
        
        storage = write_metadata(storage, storage.sample_idx, 'ris_jamming', actual_signal_power_ris, 0, ...
            noise_power, sinr_db, baseline_snr_db, effectiveness_db, current_geometry, 'basic', 1);
        
        tracking.debug_stats.ris_effectiveness = [tracking.debug_stats.ris_effectiveness, effectiveness_db];
    end
end

function [storage, tracking] = generate_basic_active_signals(CONFIG, derived_params, precomputed, storage, tracking)
    for i = 1:CONFIG.ACTIVE_JAMMING_SAMPLES
        if mod(i, max(1, floor(CONFIG.ACTIVE_JAMMING_SAMPLES/5))) == 0
            fprintf('  Active jamming progress: %d/%d\n', i, CONFIG.ACTIVE_JAMMING_SAMPLES);
        end
        
        if CONFIG.RANDOMISE_POSITIONS
            [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
        else
            x_t = 0; x_r = 10; x_i = 5; y_i = 2;
        end
        
        current_geometry = struct('x_t', x_t, 'x_r', x_r, 'x_i', x_i, 'y_i', y_i);
        
        [h_d, ~, ~, omega, PL_jammer_rx] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
            derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
            derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);
        
        tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed);
        
        tracking.debug_stats.total_active_samples = tracking.debug_stats.total_active_samples + 1;
        
        jamming_power_dbm = CONFIG.JAMMER_POWER_DBM_MIN + ...
            (CONFIG.JAMMER_POWER_DBM_MAX - CONFIG.JAMMER_POWER_DBM_MIN) * rand();
        jamming_power_tx_linear = 10^(jamming_power_dbm/10) / 1000;
        
        h_j = generate_jammer_channel(PL_jammer_rx);
        
        % Calculate theoretical powers for consistent SINR evaluation
        theoretical_signal_power = abs(h_d * omega)^2;
        theoretical_jamming_power_rx = jamming_power_tx_linear * abs(h_j)^2;
        noise_power = derived_params.sigma2;
        
        baseline_snr_db = 10*log10(theoretical_signal_power / noise_power);
        
        % Lyu et al. Equation (7): SINR = Psignal / (Pnoise + Pa)
        jammed_sinr_linear = theoretical_signal_power / (noise_power + theoretical_jamming_power_rx);
        sinr_db = 10*log10(jammed_sinr_linear);
        effectiveness_db = baseline_snr_db - sinr_db;
        
        rx_signal_clean = tx_signal * (h_d * omega);
        
        jamming_signal_baseband = sqrt(jamming_power_tx_linear) * ...
            (randn(size(tx_signal)) + 1j*randn(size(tx_signal))) / sqrt(2);
        jamming_signal_at_rx = jamming_signal_baseband * h_j;
        
        noise_signal = sqrt(noise_power/2) * ...
            (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
        
        rx_signal = rx_signal_clean + jamming_signal_at_rx + noise_signal;
        
        storage.sample_idx = storage.sample_idx + 1;
        storage.raw_signals{storage.sample_idx} = rx_signal;
        
        if mod(i, 25) == 0 || i <= 3
            fprintf('    Active jamming sample %d: LT(%.2f, 0), LR(%.2f, 0), Jammer(%.2f, %.2f), Pj=%.0fdBm, Reduction: %.1f dB\n', ...
                i, x_t, x_r, x_i, y_i, jamming_power_dbm, effectiveness_db);
        end
        
        storage = write_metadata(storage, storage.sample_idx, 'active_jamming', ...
            theoretical_signal_power, theoretical_jamming_power_rx, noise_power, ...
            sinr_db, baseline_snr_db, effectiveness_db, current_geometry, 'natural', 2);
        
        tracking.debug_stats.active_effectiveness = [tracking.debug_stats.active_effectiveness, effectiveness_db];
    end
end

function [storage, tracking] = generate_stratified_active_signals(CONFIG, derived_params, precomputed, storage, tracking)
    for band_idx = 1:length(CONFIG.ACTIVE_BAND_NAMES)
        band_name = CONFIG.ACTIVE_BAND_NAMES{band_idx};
        target_range = CONFIG.ACTIVE_JAMMING_BANDS.(band_name);

        fprintf('-Generating %s RIS jamming samples (%.1f-%.1f dB)...\n', band_name, target_range(1), target_range(2));
        
        attempts = 0;
        accepted = 0;
        target_count = CONFIG.ACTIVE_JAMMING_SAMPLES_PER_BAND;
        
        while accepted < target_count
            attempts = attempts + 1;
            tracking.debug_stats.total_active_samples = tracking.debug_stats.total_active_samples + 1;
            
            if attempts > CONFIG.MAX_ATTEMPTS_PER_BAND
                fprintf('    Warning: %s band gave up after %d attempts (got %d/%d samples)\n', ...
                    band_name, attempts, accepted, target_count);
                break;
            end
            
            if CONFIG.RANDOMISE_POSITIONS
                [x_t, x_r, x_i, y_i] = generate_random_geometry(CONFIG.X_RANGE, CONFIG.RIS_X_RANGE, CONFIG.RIS_Y_RANGE);
            else
                x_t = 0; x_r = 10; x_i = 5; y_i = 2;
            end
            
            current_geometry = struct('x_t', x_t, 'x_r', x_r, 'x_i', x_i, 'y_i', y_i);
            
            [h_d, ~, ~, omega, PL_jammer_rx] = generate_channels(x_t, x_r, x_i, y_i, CONFIG.M, CONFIG.N, ...
                derived_params.PT_linear, derived_params.A, derived_params.alpha_LT_LR, ...
                derived_params.alpha_LT_RIS, derived_params.alpha_RIS_LR);
            
            h_j = generate_jammer_channel(PL_jammer_rx);
            
            theoretical_signal_power = abs(h_d * omega)^2;
            noise_power = derived_params.sigma2;
            baseline_snr_db = 10*log10(theoretical_signal_power / noise_power);
            
            [jamming_power_dbm, jamming_power_tx_linear] = calculate_target_jamming_power(...
                target_range, baseline_snr_db, theoretical_signal_power, noise_power, h_j, CONFIG);
            
            theoretical_jamming_power_rx = jamming_power_tx_linear * abs(h_j)^2;
            jammed_sinr_linear = theoretical_signal_power / (noise_power + theoretical_jamming_power_rx);
            sinr_db = 10*log10(jammed_sinr_linear);
            effectiveness_db = baseline_snr_db - sinr_db;
            
            in_band = (effectiveness_db >= target_range(1)) && ...
                     (isinf(target_range(2)) || effectiveness_db < target_range(2));
            
            if in_band
                accepted = accepted + 1;
                
                tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed);
                
                rx_signal_clean = tx_signal * (h_d * omega);
                
                jamming_signal_baseband = sqrt(jamming_power_tx_linear) * ...
                    (randn(size(tx_signal)) + 1j*randn(size(tx_signal))) / sqrt(2);
                jamming_signal_at_rx = jamming_signal_baseband * h_j;
                
                noise_signal = sqrt(noise_power/2) * ...
                    (randn(size(rx_signal_clean)) + 1j*randn(size(rx_signal_clean)));
                
                rx_signal = rx_signal_clean + jamming_signal_at_rx + noise_signal;
                
                storage.sample_idx = storage.sample_idx + 1;
                storage.raw_signals{storage.sample_idx} = rx_signal;
                
                tracking.active_band_statistics.(band_name).accepted_count = accepted;
                tracking.active_band_statistics.(band_name).effectiveness_values = ...
                    [tracking.active_band_statistics.(band_name).effectiveness_values, effectiveness_db];
                tracking.debug_stats.active_effectiveness = [tracking.debug_stats.active_effectiveness, effectiveness_db];
                
                storage = write_metadata(storage, storage.sample_idx, 'active_jamming', ...
                    theoretical_signal_power, theoretical_jamming_power_rx, noise_power, ...
                    sinr_db, baseline_snr_db, effectiveness_db, current_geometry, band_name, 2);
                
                if mod(accepted, 25) == 0 || accepted <= 3
                    acceptance_rate = 100 * accepted / attempts;
                    fprintf('    %s sample %d/%d: LT(%.2f, 0), LR(%.2f, 0), Jammer(%.2f, %.2f), Pj: %.0fdBm, Reduction: %.1f dB (%.1f%% rate)\n', ...
                        band_name, accepted, target_count, x_t, x_r, x_i, y_i, jamming_power_dbm, effectiveness_db, acceptance_rate);
                end
            end
            
            if mod(attempts, 1000) == 0
                current_rate = 100 * accepted / attempts;
                fprintf('    %s: %d/%d after %d attempts (%.2f%% rate)\n', ...
                    band_name, accepted, target_count, attempts, current_rate);
            end
        end
        
        tracking.active_band_statistics.(band_name).total_attempts = attempts;
        final_rate = 100 * accepted / attempts;
        fprintf('  %s completed: %d samples in %d attempts (%.2f%% success rate)\n', ...
            band_name, accepted, attempts, final_rate);
    end
end

function [jamming_power_dbm, jamming_power_tx_linear] = calculate_target_jamming_power(...
    target_range, baseline_snr_db, signal_power, noise_power, h_j, CONFIG)
    
    if isinf(target_range(2))
        target_effectiveness_db = target_range(1) + 15 * rand();
    else
        target_effectiveness_db = target_range(1) + (target_range(2) - target_range(1)) * rand();
    end
    
    target_sinr_db = baseline_snr_db - target_effectiveness_db;
    target_sinr_linear = 10^(target_sinr_db/10);
    
    required_jamming_power_rx = signal_power / target_sinr_linear - noise_power;
    required_jamming_power_rx = max(required_jamming_power_rx, noise_power * 0.01);
    
    channel_gain = abs(h_j)^2;
    implementation_efficiency = 0.7 + 0.2 * rand();
    
    required_jamming_power_tx = required_jamming_power_rx / (channel_gain * implementation_efficiency);
    
    jamming_power_dbm = 10*log10(required_jamming_power_tx * 1000);
    jamming_power_dbm = max(CONFIG.JAMMER_POWER_DBM_MIN, ...
                           min(CONFIG.JAMMER_POWER_DBM_MAX, jamming_power_dbm));
    
    jamming_power_tx_linear = 10^(jamming_power_dbm/10) / 1000;
end

function h_j = generate_jammer_channel(PL_jammer_rx)
    h_j = sqrt(PL_jammer_rx/2) * (randn + 1j*randn);
    
    rician_k_factor = 0.5 + 2.5 * rand();
    
    los_component = sqrt(rician_k_factor / (1 + rician_k_factor));
    nlos_component = sqrt(1 / (1 + rician_k_factor)) * (randn + 1j*randn) / sqrt(2);
    
    h_j = h_j * (los_component + nlos_component);
    
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
    warning('Random geometry generation failed, using default');
end

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
    
    % % Uniform beamforming?
    % omega = sqrt(PT_linear/M) * ones(M, 1);

    PL_jammer_rx = PL_RIS_LR;
end

function [raw_signals, raw_metadata] = finalise_dataset(storage)
    actual_samples = storage.sample_idx;
    raw_signals = storage.raw_signals(1:actual_samples);
    raw_metadata = storage.metadata(1:actual_samples);
    
    fprintf('Final dataset: %d samples\n', actual_samples);
end

function display_generation_summary(CONFIG, raw_dataset, filename)
    fprintf('\nGENERATION COMPLETE\n');
    fprintf('File: %s\n', filename);
    fprintf('Samples: %d\n', length(raw_dataset.signals));
    
    file_info = dir(filename);
    fprintf('Size: %.1f MB\n', file_info.bytes / 1e6);
    
    scenarios = {raw_dataset.metadata.scenario};
    unique_scenarios = unique(scenarios);
    fprintf('\nDistribution:\n');
    for i = 1:length(unique_scenarios)
        count = sum(strcmp(scenarios, unique_scenarios{i}));
        fprintf('  %s: %d samples\n', unique_scenarios{i}, count);
    end
    
    ris_samples = strcmp(scenarios, 'ris_jamming');
    if any(ris_samples)
        ris_effectiveness = [raw_dataset.metadata(ris_samples).effectiveness_db];
        fprintf('\nRIS jamming effectiveness: %.1f ± %.1f dB SINR drop\n', ...
            mean(ris_effectiveness), std(ris_effectiveness));
    end
    
    active_samples = strcmp(scenarios, 'active_jamming');
    if any(active_samples)
        active_effectiveness = [raw_dataset.metadata(active_samples).effectiveness_db];
        fprintf('Active jamming effectiveness: %.1f ± %.1f dB SINR drop\n', ...
            mean(active_effectiveness), std(active_effectiveness));
    end
    
    if CONFIG.INCLUDE_ACTIVE_JAMMING
        fprintf('\nLabels: 0=normal, 1=RIS, 2=active\n');
    else
        fprintf('\nLabels: 0=normal, 1=RIS\n');
    end
    
    fprintf('\nNext step: extract features and generate features dataset\n');
    
    % Check the metadata
    table = struct2table(raw_dataset.metadata);
    mandatory = {'scenario','sinr_db','baseline_snr_db','effectiveness_db','interference_power','attack_band','label'};
    assert(all(ismember(mandatory, table.Properties.VariableNames)), 'Unified fields missing');
    assert(~any(isnan(table.effectiveness_db)), 'NaNs in effectiveness_db');
    fprintf('Metadata sanity check passed\n');
end

function storage = write_metadata(storage, idx, scenario, signal_power, interference_power, noise_power, sinr_db, baseline_snr_db, effectiveness_db, geometry, attack_band, label)

    if nargin < 12 || isempty(label)
        switch scenario
            case 'no_jamming',    label = 0;
            case 'ris_jamming',   label = 1;
            case 'active_jamming',label = 2;
            otherwise,            label = -1;
        end
    end

    storage.metadata(idx).sample_idx = idx;
    storage.metadata(idx).scenario = scenario;
    storage.metadata(idx).signal_power = signal_power;
    storage.metadata(idx).interference_power = interference_power;
    storage.metadata(idx).noise_power = noise_power;
    storage.metadata(idx).sinr_db = sinr_db;
    storage.metadata(idx).baseline_snr_db = baseline_snr_db;
    storage.metadata(idx).effectiveness_db = effectiveness_db;
    storage.metadata(idx).geometry = geometry;
    storage.metadata(idx).attack_band = attack_band;
    storage.metadata(idx).label = label; 
end

% Generate 16-QAM signal with proper constellation scaling
function tx_signal = generate_qam_signal(CONFIG, derived_params, precomputed)
    data_bits = randi([0 1], CONFIG.NUM_SYMBOLS * derived_params.BITS_PER_SYMBOL, 1);

    % Constellation normalisation
    tx_symbols = qammod(data_bits, derived_params.MODULATION_ORDER, ...
                       'InputType', 'bit', 'UnitAveragePower', true);

    % Apply pulse shaping
    tx_signal_baseband = upfirdn(tx_symbols, precomputed.RRC_FILTER, ...
                                derived_params.SAMPLES_PER_SYMBOL);
    
    % Ensure consistent signal length
    if length(tx_signal_baseband) > derived_params.SIGNAL_LENGTH
        tx_signal = tx_signal_baseband(1:derived_params.SIGNAL_LENGTH);
    elseif length(tx_signal_baseband) < derived_params.SIGNAL_LENGTH
        padding_length = derived_params.SIGNAL_LENGTH - length(tx_signal_baseband);
        tx_signal = [tx_signal_baseband; zeros(padding_length, 1)];
    else
        tx_signal = tx_signal_baseband;
    end
end