function generate_features_dataset_research(raw_signals_file)
    % Compatible with generate_raw_signals_stratified.m output
    % Calls extract_features_research.m for feature extraction
    close all;

    if nargin < 1
        % training dataset file
        % raw_signals_file = 'signals/raw_signals_stratified_seed42_train.mat';

        % For example generating test dataset:
        raw_signals_file = 'signals/raw_signals_stratified_seed123_stealthy_test.mat';
        if ~exist(raw_signals_file, 'file')
            error('Raw signals file not found. Check file path or generate raw signals first via generate_raw_signals_stratified.m');
        end
    end
    
    fprintf('DATASET GENERATOR \n');
    fprintf('Loading raw signals: %s\n', raw_signals_file);
    
    if ~exist(raw_signals_file, 'file')
        error('Raw signals file not found: %s', raw_signals_file);
    end
    
    load(raw_signals_file, 'raw_dataset');
    % NOTE the output directory, for now output to 'dataset_output' folder within matlab folder so it doesn't override pre-generated dataset
    output_dir = 'dataset_output/';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Can edit file name
    csv_filename = fullfile(output_dir, 'jamming_features_research.csv');
    
    signals = raw_dataset.signals;
    metadata = raw_dataset.metadata;
    CONFIG = raw_dataset.config;
    derived_params = raw_dataset.derived_params;
    
    fprintf('\nValidating metadata structure:\n');
    if isstruct(metadata) && ~isempty(metadata)
        metadata_fields = fieldnames(metadata(1));
        fprintf('  Available metadata fields: %d\n', length(metadata_fields));
        
        effectiveness_field = '';
        possible_names = {'effectiveness_db', 'power_reduction_db', 'sinr_reduction_db', 'jamming_effectiveness'};
        
        for name_idx = 1:length(possible_names)
            if isfield(metadata(1), possible_names{name_idx})
                effectiveness_field = possible_names{name_idx};
                break;
            end
        end
        
        if isempty(effectiveness_field)
            fprintf(' No effectiveness field found, will calculate from SINR\n');
        else
            fprintf(' Using effectiveness field: %s\n', effectiveness_field);
        end
    else
        error('Metadata structure is invalid or empty');
    end
    
    feature_names = {...
        'sinr_estimate', 'mean_magnitude', 'std_magnitude', 'peak_to_avg_ratio', 'received_power', ...
        'mean_psd_db', 'std_psd_db', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_entropy', 'spectral_flatness', ...
        'mean_real', 'mean_imag', 'iq_power_ratio_db', 'amplitude_kurtosis'
    };

    % Verify feature count are matched up
    expected_features = 16;
    if length(feature_names) ~= expected_features
        error('Feature names count (%d) does not match expected features (%d)', ...
            length(feature_names), expected_features);
    end
        
    fprintf('\nFeature domains:\n');
    fprintf('  Category 1 (Power): features 1-5\n');
    fprintf('  Category 2 (Spectral): features 6-12\n');
    fprintf('  Category 3 (Statistical): features 13-16\n');
    
    fprintf('Loaded %d raw signals\n', length(signals));
    if isfield(raw_dataset, 'stratification_type')
        fprintf('Configuration: %s\n', raw_dataset.stratification_type);
    end
    
    fprintf('\nExtracting feature parameters...\n');
    if isfield(raw_dataset, 'feature_extraction_params')
        feature_params = raw_dataset.feature_extraction_params;
        fprintf('  Found feature parameters\n');
    else
        fprintf('  Using fallback feature parameters\n');
        feature_params = struct();
        feature_params.fs = CONFIG.SAMPLE_RATE;
        feature_params.fft_size = derived_params.FEATURE_FFT_SIZE;
    end
    
    fprintf('  Sample rate: %.0f Hz\n', feature_params.fs);
    fprintf('  FFT size: %d\n', feature_params.fft_size);
        
    n_samples = length(signals);
    n_features = length(feature_names);
    features = zeros(n_samples, n_features);
    
    fprintf('Extracting %d research features from %d signals...\n', n_features, n_samples);
    
    extraction_start_time = tic;
    feature_extraction_errors = 0;
    
    for i = 1:n_samples
        if mod(i, max(1, floor(n_samples/20))) == 0
            elapsed_time = toc(extraction_start_time);
            estimated_total = elapsed_time * n_samples / i;
            remaining_time = estimated_total - elapsed_time;
            
            fprintf('  Progress: %d/%d (%.1f%%) - ETA: %.1f seconds\n', ...
                i, n_samples, 100*i/n_samples, remaining_time);
        end
        
        rx_signal = signals{i};
        
        thermal_noise_power = metadata(i).noise_power;
        
        if isfield(metadata(i), 'interference_power')
            interference_power = metadata(i).interference_power;
        else
            interference_power = 0;
        end
        
        effective_noise_power = thermal_noise_power + interference_power;
        
        if ~isnumeric(rx_signal) || isempty(rx_signal)
            warning('Invalid signal format at sample %d, filling with zeros', i);
            features(i, :) = zeros(1, n_features);
            feature_extraction_errors = feature_extraction_errors + 1;
            continue;
        end
        
        rx_signal = rx_signal(:);
        
        fs = feature_params.fs;
        fft_size = feature_params.fft_size;
        
        try
            feature_row = extract_features_research(rx_signal, effective_noise_power, fs, fft_size);
            
            if size(feature_row, 2) ~= n_features
                error('Feature extractor returned wrong dimensions: [%d x %d], expected [1 x %d]', ...
                    size(feature_row, 1), size(feature_row, 2), n_features);
            end
            
            % Override SINR with metadata value for accuracy
            if isfield(metadata(i), 'sinr_db')
                feature_row(1) = max(-50, min(80, metadata(i).sinr_db));
            end
            
            features(i, :) = feature_row;
            
        catch ME
            fprintf('Error in feature extraction for sample %d (%s): %s\n', ...
                i, metadata(i).scenario, ME.message);
            features(i, :) = zeros(1, n_features);
            feature_extraction_errors = feature_extraction_errors + 1;
        end
        
        if any(~isfinite(features(i, :)))
            nan_count = sum(~isfinite(features(i, :)));
            warning('Non-finite features in sample %d (%s): %d features', ...
                i, metadata(i).scenario, nan_count);
            features(i, ~isfinite(features(i, :))) = 0;
        end
    end
    
    extraction_time = toc(extraction_start_time);
    fprintf('Research feature extraction complete (%.1f seconds)\n', extraction_time);
    fprintf('Extracted %d features per sample\n', n_features);
    
    if feature_extraction_errors > 0
        fprintf('Warning: %d feature extraction errors occurred (%.1f%% of samples)\n', ...
            feature_extraction_errors, 100*feature_extraction_errors/n_samples);
    end
    
    % Generate labels from metadata
    labels = zeros(n_samples, 1);
    for i = 1:n_samples
        if isfield(metadata(i), 'label') && ~isempty(metadata(i).label)
            labels(i) = metadata(i).label;
        else
            switch metadata(i).scenario
                case 'no_jamming'
                    labels(i) = 0;
                case 'ris_jamming'
                    labels(i) = 1;
                case 'active_jamming'
                    labels(i) = 2;
                otherwise
                    warning('Unknown scenario: %s', metadata(i).scenario);
                    labels(i) = -1;
            end
        end
    end
    

    fprintf('\n RESEARCH FEATURE VALIDATION \n');
    
    zero_features = sum(abs(features) < 1e-12, 1);
    constant_features = std(features, 0, 1) < 1e-10;
    nan_features = sum(~isfinite(features), 1);
    
    fprintf('Feature quality analysis:\n');
    if any(constant_features)
        fprintf('  Constant features: %d\n', sum(constant_features));
        constant_indices = find(constant_features);
        for idx = constant_indices(1:min(5, end))
            fprintf('    %s (index %d): value = %.6f\n', feature_names{idx}, idx, features(1, idx));
        end
    else
        fprintf('  No constant features detected\n');
    end
    
    problematic_features = find(zero_features > n_samples * 0.5);
    if ~isempty(problematic_features)
        fprintf('  High-zero features (>50%% zeros): %d\n', length(problematic_features));
        for idx = problematic_features(1:min(3, end))
            fprintf('    %s: %.1f%% zeros\n', feature_names{idx}, 100*zero_features(idx)/n_samples);
        end
    else
        fprintf('  No high-zero features detected\n');
    end
    
    if any(nan_features > 0)
        fprintf('  NaN/Inf features: %d (cleaned)\n', sum(nan_features > 0));
    else
        fprintf('  No NaN/Inf features detected\n');
    end
    
    fprintf('\n CORRELATION ANALYSIS \n');
    
    non_constant_features = features(:, ~constant_features);
    non_constant_names = feature_names(~constant_features);
    
    if size(non_constant_features, 2) > 1
        corr_matrix = corrcoef(non_constant_features);
        corr_matrix(isnan(corr_matrix)) = 0;
        
        high_corr_threshold = 0.95;
        [high_i, high_j] = find(abs(corr_matrix) > high_corr_threshold & corr_matrix ~= 1);
        
        unique_pairs = 0;
        if ~isempty(high_i)
            for k = 1:length(high_i)
                if high_i(k) < high_j(k)
                    unique_pairs = unique_pairs + 1;
                end
            end
        end
        
        fprintf('High correlation analysis (|r| > %.2f): %d pairs\n', high_corr_threshold, unique_pairs);
        
        if unique_pairs > 0 && unique_pairs <= 15
            fprintf('  High correlation pairs:\n');
            pair_count = 0;
            for k = 1:length(high_i)
                i = high_i(k);
                j = high_j(k);
                if i < j
                    pair_count = pair_count + 1;
                    if pair_count <= 10
                        corr_val = corr_matrix(i, j);
                        fprintf('    %s <-> %s: r = %.3f\n', ...
                            non_constant_names{i}, non_constant_names{j}, corr_val);
                    end
                end
            end
            if pair_count > 10
                fprintf('    ... and %d more pairs\n', pair_count - 10);
            end
            fprintf('  Consider feature selection to remove redundancy\n');
        else
            fprintf('  Minimal feature redundancy detected\n');
        end
        
        imagesc(corr_matrix);
        colorbar;
        colormap(parula);
        clim([-1, 1]);
        
        title('Research-Grade 18-Feature RIS Dataset Correlation Matrix', 'FontSize', 16, 'FontWeight', 'bold');
        xlabel('Feature Index', 'FontSize', 14);
        ylabel('Feature Index', 'FontSize', 14);
        
        category_boundaries = [5, 12, 16];
        colours = {'white', 'red', 'yellow', 'cyan'};
        
        for b = 1:length(category_boundaries)
            boundary = category_boundaries(b);
            if boundary <= size(corr_matrix, 1)
                line([boundary+0.5, boundary+0.5], [0.5, size(corr_matrix,1)+0.5], ...
                     'Color', colours{min(b, length(colours))}, 'LineWidth', 2);
                line([0.5, size(corr_matrix,2)+0.5], [boundary+0.5, boundary+0.5], ...
                     'Color', colours{min(b, length(colours))}, 'LineWidth', 2);
            end
        end
        
        category_positions = [3, 9, 14.5, 17.5];
        category_short_labels = {'Power', 'Spectral', 'Statistical', 'RIS'};
        
        for p = 1:length(category_positions)
            if category_positions(p) <= size(corr_matrix, 1)
                text(category_positions(p), 1, category_short_labels{p}, ...
                    'Color', 'white', 'FontWeight', 'bold', 'FontSize', 10, ...
                    'HorizontalAlignment', 'center');
            end
        end
        
        grid on;
        set(gca, 'GridAlpha', 0.3);
    else
        fprintf('Insufficient non-constant features for correlation analysis\n');
    end
    
    fprintf('\n DATASET ANALYSIS \n');
    
    fprintf('Research dataset statistics:\n');
    fprintf('  Total samples: %d\n', n_samples);
    fprintf('  Features: %d (streamlined set)\n', n_features);
    fprintf('  Extraction time: %.1f seconds (%.2f ms/sample)\n', ...
        extraction_time, 1000*extraction_time/n_samples);
    
    unique_labels = unique(labels);
    
    for i = 1:length(unique_labels)
        count = sum(labels == unique_labels(i));
        percentage = 100 * count / n_samples;
        
        switch unique_labels(i)
            case 0
                class_name = 'no_jamming';
            case 1
                class_name = 'ris_jamming';
            case 2
                class_name = 'active_jamming';
            otherwise
                class_name = sprintf('unknown_%d', unique_labels(i));
        end
        
        fprintf(' %s: %d samples (%.1f%%)\n', class_name, count, percentage);
    end
        
    % Extract band information for CSV
    band_name_array = cell(n_samples, 1);
    for i = 1:n_samples
        band_name_array{i} = metadata(i).attack_band;
    end
    
    % Display band distribution
    fprintf('Band distribution summary:\n');
    unique_bands = unique(band_name_array);
    for i = 1:length(unique_bands)
        count = sum(strcmp(band_name_array, unique_bands{i}));
        fprintf('  %s: %d samples (%.1f%%)\n', unique_bands{i}, count, 100*count/n_samples);
    end
    
    if size(labels, 2) > size(labels, 1)
        labels = labels';
    end
    
    try
        csv_data = array2table([features, labels], 'VariableNames', [feature_names, {'label'}]);
        csv_data.band_name = band_name_array;
        writetable(csv_data, csv_filename);
        fprintf('CSV saved: %s\n', csv_filename);
    catch ME
        fprintf('Warning: Could not save CSV to specified location: %s\n', ME.message);
        try
            csv_data = array2table([features, labels], 'VariableNames', [feature_names, {'label'}]);
            csv_data.band_name = band_name_array;
            writetable(csv_data, csv_filename);
            fprintf('CSV saved to current directory: %s\n', csv_filename);
        catch ME2
            fprintf('Warning: CSV save failed: %s\n', ME2.message);
        end
    end
    
    
    if CONFIG.INCLUDE_ACTIVE_JAMMING
        fprintf('Labels: 0=normal, 1=RIS-jamming, 2=active-jamming\n');
    else
        fprintf('Binary detection: 0=normal, 1=RIS-jamming\n');
    end
    
    fprintf('  CSV file in: %s\n', csv_filename);
    
    fprintf('Dataset generation complete\n');
end
