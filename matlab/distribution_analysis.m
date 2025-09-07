%% Load and analyse dataset distributions from given .mat file
%% Author: Hui Shing
% This script loads an arbitrary .mat file (containing raw signals samples and metadata)
% Produces figures for class distribution, effectiveness band distributions and geometry distributions analysis
% 3 plots tabs should pop up
% Run to see the distribution analysis for the raw signals used for training.
clear; close all;

% Load dataset
dataset_file = 'signals/raw_signals.mat';
load(dataset_file, 'raw_dataset');

fprintf('DISTRIBUTION ANALYSIS FROM .MAT\n');

%% Extract effectiveness data from metadata
scenarios = {raw_dataset.metadata.scenario};
ris_samples = strcmp(scenarios, 'ris_jamming');
active_samples = strcmp(scenarios, 'active_jamming');
no_jam_samples = strcmp(scenarios, 'no_jamming');

% RIS effectiveness using effectiveness_db field
if any(ris_samples)
    ris_effectiveness = [raw_dataset.metadata(ris_samples).effectiveness_db];
else
    ris_effectiveness = [];
end

% Active effectiveness using effectiveness_db field
if any(active_samples)
    active_effectiveness = [raw_dataset.metadata(active_samples).effectiveness_db];
else
    active_effectiveness = [];
end

%% Create comprehensive distribution plots
figure('Position', [100, 100, 1400, 800]);

% Subplot 1: RIS effectiveness histogram
subplot(2, 3, 1);
if ~isempty(ris_effectiveness)
    histogram(ris_effectiveness, 20, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'black', 'FaceAlpha', 0.7);

    title('RIS Attack Effectiveness Distribution');
    xlabel('SINR Reduction (dB)');
    ylabel('Count');
    grid on;
    
    % Add statistics text
    text(0.60, 0.95, sprintf('μ = %.1f dB\nσ = %.1f dB\nRange: %.1f - %.1f', ...
         mean(ris_effectiveness), std(ris_effectiveness), ...
         min(ris_effectiveness), max(ris_effectiveness)), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10, ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end

% Subplot 2: Active effectiveness histogram
subplot(2, 3, 2);
if ~isempty(active_effectiveness)
    histogram(active_effectiveness, 20, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'black', 'FaceAlpha', 0.7);
    title('Active Jamming Effectiveness Distribution');
    xlabel('SINR Reduction (dB)');
    ylabel('Count');
    grid on;
    
    text(0.60, 0.95, sprintf('μ = %.1f dB\nσ = %.1f dB\nRange: %.1f - %.1f', ...
         mean(active_effectiveness), std(active_effectiveness), ...
         min(active_effectiveness), max(active_effectiveness)), ...
         'Units', 'normalized', 'VerticalAlignment', 'top', 'FontSize', 10, ...
         'BackgroundColor', 'white', 'EdgeColor', 'black');
end

% Subplot 3: Combined effectiveness comparison
subplot(2, 3, 3);
hold on;
if ~isempty(ris_effectiveness)
    histogram(ris_effectiveness, 15, 'FaceColor', [0.2, 0.4, 0.8], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
end
if ~isempty(active_effectiveness)
    histogram(active_effectiveness, 15, 'FaceColor', [0.8, 0.2, 0.2], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
end
title('Attack Effectiveness Comparison');
xlabel('Effectiveness (dB)');
ylabel('Count');
legend({'RIS Jamming', 'Active Jamming'}, 'Location', 'best');
grid on;

% Subplot 4: Cumulative distributions
subplot(2, 3, 4);
hold on;
if ~isempty(ris_effectiveness)
    [ris_cdf_y, ris_cdf_x] = ecdf(ris_effectiveness);
    plot(ris_cdf_x, ris_cdf_y, 'b-', 'LineWidth', 2);
end
if ~isempty(active_effectiveness)
    [active_cdf_y, active_cdf_x] = ecdf(active_effectiveness);
    plot(active_cdf_x, active_cdf_y, 'r-', 'LineWidth', 2);
end
title('Cumulative Distribution Functions');
xlabel('Effectiveness (dB)');
ylabel('Cumulative Probability');
legend({'RIS', 'Active'}, 'Location', 'best');
grid on;

% Subplot 5: Sample counts per scenario
subplot(2, 3, 5);
scenario_counts = [sum(no_jam_samples), sum(ris_samples), sum(active_samples)];
scenario_labels = {'No Jamming', 'RIS Jamming', 'Active Jamming'};
colours = [0.5, 0.7, 0.5; 0.2, 0.4, 0.8; 0.8, 0.2, 0.2];
bar(scenario_counts, 'FaceColor', 'flat', 'CData', colours, 'EdgeColor', 'black');
set(gca, 'XTickLabel', scenario_labels, 'XTickLabelRotation', 45);
title('Class Distribution');
ylabel('Count');
grid on;

% Subplot 6: Box plot comparison
subplot(2, 3, 6);
if ~isempty(ris_effectiveness) && ~isempty(active_effectiveness)
    effectiveness_data = [ris_effectiveness'; active_effectiveness'];
    group_labels = [ones(length(ris_effectiveness), 1); 2*ones(length(active_effectiveness), 1)];
    boxplot(effectiveness_data, group_labels, 'Labels', {'RIS', 'Active'});
    title('Effectiveness Box Plot Comparison');
    ylabel('Effectiveness (dB)');
    grid on;
elseif ~isempty(ris_effectiveness)
    % RIS only
    boxplot(ris_effectiveness, 'Labels', {'RIS'});
    title('RIS Effectiveness Distribution');
    ylabel('Effectiveness (dB)');
    grid on;
end

%  title
sgtitle('RIS Jamming Dataset - Distribution Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Extract SINR values for additional analysis
sinr_values = [raw_dataset.metadata.sinr_db];
baseline_snr_values = [raw_dataset.metadata.baseline_snr_db];

% Plot SINR analysis
figure('Position', [150, 50, 1200, 800]);

% SINR distributions by scenario
subplot(2, 2, 1);
hold on;
if any(no_jam_samples)
    no_jam_sinr = sinr_values(no_jam_samples);
    histogram(no_jam_sinr, 15, 'FaceColor', [0.5, 0.7, 0.5], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
end
if any(ris_samples)
    ris_sinr = sinr_values(ris_samples);
    histogram(ris_sinr, 15, 'FaceColor', [0.2, 0.4, 0.8], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
end
if any(active_samples)
    active_sinr = sinr_values(active_samples);
    histogram(active_sinr, 15, 'FaceColor', [0.8, 0.2, 0.2], 'FaceAlpha', 0.6, 'EdgeColor', 'none');
end
title('SINR Distributions by Scenario');
xlabel('SINR (dB)');
ylabel('Count');
legend(scenario_labels(scenario_counts > 0), 'Location', 'best');
grid on;

% Baseline vs achieved SINR scatter
subplot(2, 2, 2);
colours_map = zeros(length(scenarios), 3);
colours_map(no_jam_samples, :) = repmat([0.5, 0.7, 0.5], sum(no_jam_samples), 1);
colours_map(ris_samples, :) = repmat([0.2, 0.4, 0.8], sum(ris_samples), 1);
colours_map(active_samples, :) = repmat([0.8, 0.2, 0.2], sum(active_samples), 1);

scatter(baseline_snr_values, sinr_values, 30, colours_map, 'filled', 'MarkerEdgeColor', 'black');
hold on;
plot([min(baseline_snr_values), max(baseline_snr_values)], ...
     [min(baseline_snr_values), max(baseline_snr_values)], 'k--', 'LineWidth', 1.5);
title('Baseline vs Achieved SINR');
xlabel('Baseline SINR (dB)');
ylabel('Achieved SINR (dB)');
grid on;
axis equal;

% Effectiveness vs baseline SNR
subplot(2, 2, 3);
all_effectiveness = [raw_dataset.metadata.effectiveness_db];
scatter(baseline_snr_values, all_effectiveness, 30, colours_map, 'filled', 'MarkerEdgeColor', 'black');
title('Attack Effectiveness vs Baseline SINR');
xlabel('Baseline SINR (dB)');
ylabel('Effectiveness (dB)');
grid on;

% Band analysis if stratified
subplot(2, 2, 4);
if isfield(raw_dataset.config, 'USE_STRATIFICATION') && raw_dataset.config.USE_STRATIFICATION
    attack_bands = {raw_dataset.metadata.attack_band};
    unique_bands = unique(attack_bands(~strcmp(attack_bands, 'none') & ~strcmp(attack_bands, '')));
    
    if ~isempty(unique_bands)
        hold on;
        band_colours = lines(length(unique_bands));
        
        for i = 1:length(unique_bands)
            band_name = unique_bands{i};
            band_mask = strcmp(attack_bands, band_name);
            if any(band_mask)
                band_effectiveness = all_effectiveness(band_mask);
                histogram(band_effectiveness, 10, 'FaceColor', band_colours(i, :), ...
                         'FaceAlpha', 0.6, 'EdgeColor', 'black');
            end
        end
        
        title('Effectiveness by Attack Band');
        xlabel('Effectiveness (dB)');
        ylabel('Count');
        legend(unique_bands, 'Location', 'best');
        grid on;
    else
        text(0.5, 0.5, 'No stratification bands found', 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', 'Units', 'normalized', 'FontSize', 12);
        title('Band Analysis');
    end
else
    text(0.5, 0.5, 'Stratification not used', 'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', 'Units', 'normalized', 'FontSize', 12);
    title('Band Analysis');
end

sgtitle('SINR and Effectiveness Analysis', 'FontSize', 14, 'FontWeight', 'bold');

%% Statistical analysis
fprintf('\nSTATISTICAL ANALYSIS:\n');
fprintf('Sample distribution:\n');
fprintf('  No jamming: %d samples\n', sum(no_jam_samples));
fprintf('  RIS jamming: %d samples\n', sum(ris_samples));
fprintf('  Active jamming: %d samples\n', sum(active_samples));

if ~isempty(ris_effectiveness)
    fprintf('\nRIS attack effectiveness:\n');
    fprintf('  Mean: %.1f dB\n', mean(ris_effectiveness));
    fprintf('  Std: %.1f dB\n', std(ris_effectiveness));
    fprintf('  Range: %.1f to %.1f dB\n', min(ris_effectiveness), max(ris_effectiveness));
    fprintf('  Median: %.1f dB\n', median(ris_effectiveness));
    
end

if ~isempty(active_effectiveness)
    fprintf('\nActive jamming effectiveness:\n');
    fprintf('  Mean: %.1f dB\n', mean(active_effectiveness));
    fprintf('  Std: %.1f dB\n', std(active_effectiveness));
    fprintf('  Range: %.1f to %.1f dB\n', min(active_effectiveness), max(active_effectiveness));
    fprintf('  Median: %.1f dB\n', median(active_effectiveness));
end

% SINR analysis
fprintf('\nSINR analysis:\n');
if any(no_jam_samples)
    no_jam_sinr = sinr_values(no_jam_samples);
    fprintf('  No jamming SINR: %.1f ± %.1f dB\n', mean(no_jam_sinr), std(no_jam_sinr));
end

if any(ris_samples)
    ris_sinr = sinr_values(ris_samples);
    fprintf('  RIS jamming SINR: %.1f ± %.1f dB\n', mean(ris_sinr), std(ris_sinr));
end

if any(active_samples)
    active_sinr = sinr_values(active_samples);
    fprintf('  Active jamming SINR: %.1f ± %.1f dB\n', mean(active_sinr), std(active_sinr));
end

% Baseline SNR statistics (no jam)
fprintf('\nBaseline SNR statistics:\n');
fprintf('  Overall baseline: %.1f ± %.1f dB\n', mean(baseline_snr_values), std(baseline_snr_values));

%% Band effectiveness analysis if stratified
if isfield(raw_dataset.config, 'USE_STRATIFICATION') && raw_dataset.config.USE_STRATIFICATION
    fprintf('\nStratification band analysis:\n');
    attack_bands = {raw_dataset.metadata.attack_band};
    unique_bands = unique(attack_bands(~strcmp(attack_bands, 'none') & ~strcmp(attack_bands, '')));
    
    for i = 1:length(unique_bands)
        band_name = unique_bands{i};
        band_mask = strcmp(attack_bands, band_name);
        if any(band_mask)
            band_effectiveness = all_effectiveness(band_mask);
            fprintf('  %s band: %.1f ± %.1f dB (n=%d)\n', ...
                band_name, mean(band_effectiveness), std(band_effectiveness), length(band_effectiveness));
        end
    end
end

%% Geometry analysis
if isfield(raw_dataset.metadata, 'geometry')
    fprintf('\n--- DETAILED GEOMETRY ANALYSIS ---\n');
    geometries = [raw_dataset.metadata.geometry];
    if ~isempty(geometries)
        x_t_values = [geometries.x_t];
        x_r_values = [geometries.x_r];
        x_i_values = [geometries.x_i];
        y_i_values = [geometries.y_i];
        
        % Calculate derived metrics
        lr_separations = x_r_values - x_t_values;  % TX-RX distance
        ris_x_fractions = (x_i_values - x_t_values) ./ lr_separations;  % RIS position fraction
        
        % Distance calculations
        d_lt_ris = sqrt((x_i_values - x_t_values).^2 + y_i_values.^2);
        d_ris_lr = sqrt((x_r_values - x_i_values).^2 + y_i_values.^2);
        d_lt_lr = abs(x_r_values - x_t_values);
        path_ratios = (d_lt_ris + d_ris_lr) ./ d_lt_lr;
        
        fprintf('Transmitter positions (x_t):\n');
        fprintf('  Range: %.2f to %.2f m\n', min(x_t_values), max(x_t_values));
        fprintf('  Mean ± Std: %.2f ± %.2f m\n', mean(x_t_values), std(x_t_values));
        
        fprintf('\nReceiver positions (x_r):\n');
        fprintf('  Range: %.2f to %.2f m\n', min(x_r_values), max(x_r_values));
        fprintf('  Mean ± Std: %.2f ± %.2f m\n', mean(x_r_values), std(x_r_values));
        
        fprintf('\nRIS X positions (x_i):\n');
        fprintf('  Range: %.2f to %.2f m\n', min(x_i_values), max(x_i_values));
        fprintf('  Mean ± Std: %.2f ± %.2f m\n', mean(x_i_values), std(x_i_values));
        
        fprintf('\nRIS Y positions (y_i):\n');
        fprintf('  Range: %.2f to %.2f m\n', min(y_i_values), max(y_i_values));
        fprintf('  Mean ± Std: %.2f ± %.2f m\n', mean(y_i_values), std(y_i_values));
        
        fprintf('\nTX-RX separations:\n');
        fprintf('  Range: %.2f to %.2f m\n', min(lr_separations), max(lr_separations));
        fprintf('  Mean ± Std: %.2f ± %.2f m\n', mean(lr_separations), std(lr_separations));
        
        fprintf('\nRIS positioning fractions (along TX-RX line):\n');
        fprintf('  Range: %.3f to %.3f\n', min(ris_x_fractions), max(ris_x_fractions));
        fprintf('  Mean ± Std: %.3f ± %.3f\n', mean(ris_x_fractions), std(ris_x_fractions));
        fprintf('  (0.0 = at TX, 1.0 = at RX, 0.5 = midpoint)\n');
        

        % Geometry diversity check
        unique_positions = length(unique(round([x_i_values; y_i_values]', 2), 'rows'));
        total_samples = length(x_i_values);
        diversity_ratio = unique_positions / total_samples;
        
        fprintf('\nGeometry diversity:\n');
        fprintf('  Unique RIS positions: %d/%d samples (%.1f%% diversity)\n', ...
            unique_positions, total_samples, diversity_ratio * 100);
        
        if diversity_ratio < 0.5
            fprintf('  WARNING: Low geometry diversity - many samples may be using fallback position!\n');
        end
        
        % Check for fallback usage (default: x_i=5, y_i=2)
        fallback_mask = abs(x_i_values - 5) < 0.1 & abs(y_i_values - 2) < 0.1;
        fallback_count = sum(fallback_mask);
        fallback_rate = fallback_count / total_samples * 100;
        
        fprintf('\nFallback position usage:\n');
        fprintf('  Samples at default (5, 2): %d/%d (%.1f%%)\n', ...
            fallback_count, total_samples, fallback_rate);
        
        if fallback_rate > 20
            fprintf('  WARNING: High fallback usage suggests geometry generation is frequently failing!\n');
        end
    end
end

%% Add geometry visualisation
if exist('geometries', 'var') && ~isempty(geometries)
    figure('Position', [200, 100, 1000, 600]);
    
    % Subplot 1: Geometry scatter plot
    subplot(2, 3, 1);
    scatter(x_i_values, y_i_values, 30, 'filled', 'MarkerFaceColor', [0.2, 0.4, 0.8], 'MarkerEdgeColor', 'black');
    xlabel('RIS X Position (m)');
    ylabel('RIS Y Position (m)');
    title('RIS Position Distribution');
    grid on;
    axis equal;
    
    % Highlight fallback positions
    hold on;
    fallback_mask = abs(x_i_values - 5) < 0.1 & abs(y_i_values - 2) < 0.1;
    if any(fallback_mask)
        scatter(x_i_values(fallback_mask), y_i_values(fallback_mask), 60, 'r', 'filled', 'MarkerEdgeColor', 'black');
        legend({'Generated positions', 'Fallback positions'}, 'Location', 'best');
    end
    
    % Subplot 2: TX-RX separation distribution
    subplot(2, 3, 2);
    histogram(lr_separations, 15, 'FaceColor', [0.5, 0.7, 0.5], 'EdgeColor', 'black');
    xlabel('TX-RX Separation (m)');
    ylabel('Count');
    title('TX-RX Distance Distribution');
    grid on;
    
    % Subplot 3: RIS positioning fractions
    subplot(2, 3, 3);
    histogram(ris_x_fractions, 15, 'FaceColor', [0.8, 0.6, 0.2], 'EdgeColor', 'black');
    xlabel('RIS Position Fraction');
    ylabel('Count');
    title('RIS X-Position Fraction Distribution');
    grid on;
    
    % Subplot 4: Path ratio distribution
    subplot(2, 3, 4);
    histogram(path_ratios, 20, 'FaceColor', [0.6, 0.2, 0.8], 'EdgeColor', 'black');
    xlabel('Path Ratio (RIS/Direct)');
    ylabel('Count');
    title('Path Length Ratio Distribution');
    grid on;
    
    % Add vertical line at 1.1 constraint
    hold on;
    ylims = ylim;
    plot([1.1, 1.1], ylims, 'r--', 'LineWidth', 2);
    text(0.1, ylims(2)*0.9, 'Current constraint limit', 'Rotation', 90, 'VerticalAlignment', 'bottom');
    
    % Subplot 5: Y position distribution
    subplot(2, 3, 5);
    histogram(y_i_values, 15, 'FaceColor', [0.2, 0.8, 0.6], 'EdgeColor', 'black');
    xlabel('RIS Y Position (m)');
    ylabel('Count');
    title('RIS Y-Position Distribution');
    grid on;
    
    % Subplot 6: Geometry effectiveness correlation
    subplot(2, 3, 6);
    if exist('ris_effectiveness', 'var') && ~isempty(ris_effectiveness)
        ris_mask = strcmp({raw_dataset.metadata.scenario}, 'ris_jamming');
        if any(ris_mask)
            ris_geometries = geometries(ris_mask);
            ris_y_positions = [ris_geometries.y_i];
            
            scatter(ris_y_positions, ris_effectiveness, 30, 'filled', 'MarkerFaceColor', [0.2, 0.4, 0.8], 'MarkerEdgeColor', 'black');
            xlabel('RIS Y Position (m)');
            ylabel('Attack Effectiveness (dB)');
            title('Geometry vs Effectiveness');
            grid on;
            
            % Add correlation coefficient
            if length(ris_y_positions) == length(ris_effectiveness)
                [r, p] = corrcoef(ris_y_positions, ris_effectiveness);
                text(0.05, 0.95, sprintf('r = %.3f\np = %.3f', r(1,2), p(1,2)), ...
                     'Units', 'normalized', 'VerticalAlignment', 'top', ...
                     'BackgroundColor', 'white', 'EdgeColor', 'black');
            end
        end
    else
        text(0.5, 0.5, 'RIS effectiveness data not available', 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', 'Units', 'normalized');
        title('Geometry vs Effectiveness');
    end
    
    sgtitle('Geometry Analysis Visualisation', 'FontSize', 14, 'FontWeight', 'bold');
    
    saveas(gcf, 'geometry_analysis.png');
    fprintf('\nGeometry analysis plot saved: geometry_analysis.png\n');
end

saveas(figure(1), 'distribution_analysis.png');
saveas(figure(2), 'sinr_effectiveness_analysis.png');
