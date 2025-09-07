function [features, feature_names] = extract_features_research(rx_signal, noise_power, fs, fft_size)
    
    if nargin < 4, fft_size = 4096; end
    
    rx_signal = rx_signal(:);

    signal_length = length(rx_signal);
    
    if signal_length < 64
        rx_signal = [rx_signal; zeros(64 - signal_length, 1)];
        signal_length = 64;
    end
    
    features = zeros(1, 16);
    
    signal_power = mean(abs(rx_signal).^2);
    amplitude = abs(rx_signal);
    real_part = real(rx_signal);
    imag_part = imag(rx_signal);
    
    if signal_power <= eps
        amplitude_sorted = sort(amplitude);
        noise_floor_idx = max(1, floor(0.1 * length(amplitude_sorted)));
        signal_peak_idx = max(1, floor(0.9 * length(amplitude_sorted)));
        noise_floor = mean(amplitude_sorted(1:noise_floor_idx));
        signal_peak = mean(amplitude_sorted(signal_peak_idx:end));
        signal_power = signal_peak^2;
    end
    if noise_power <= 0
        noise_power = eps;
    end
    
    % One-sided PSD (proper windowing + scaling)
    N_fft = min([fft_size, signal_length]);
    psd = [];
    
    if signal_length >= N_fft
        if exist('hann','builtin') || exist('hann','file')
            w = hann(N_fft);
        else
            w = ones(N_fft,1);
        end
        xw = rx_signal(1:N_fft) .* w;
        U = (1/N_fft) * sum(w.^2);
        X = fft(xw, N_fft);
        P2 = (abs(X).^2) / (fs * N_fft * U);
        
        psd = P2(1:floor(N_fft/2)+1);
        if numel(psd) > 2
            psd(2:end-1) = 2*psd(2:end-1);
        end
    end
    
    % Category 1: Power Analysis (Features 1-5)
    sinr_estimate = 10*log10(signal_power / (noise_power + eps));
    features(1) = sinr_estimate;
    
    features(2) = mean(amplitude);
    
    features(3) = std(amplitude);
    
    peak_power = max(amplitude.^2);
    avg_power = mean(amplitude.^2);
    features(4) = 10*log10((peak_power + eps) / (avg_power + eps));
    
    features(5) = 10*log10(signal_power + eps);
    
    % Category 2: Spectral Analysis (Features 6-12)
    if ~isempty(psd)
        psd_log = 10*log10(psd + eps);
        
        features(6) = mean(psd_log);
        
        features(7) = std(psd_log);
        
        features(8) = calculate_spectral_centroid(psd, fs);
        
        features(9) = calculate_spectral_rolloff(psd, fs, 0.90);
        
        p = psd(:); c = cumsum(p); T = c(end);
        if T > eps
            f = linspace(0, fs/2, numel(p))';
            f_lo = f(find(c >= 0.05*T, 1, 'first'));
            f_hi = f(find(c >= 0.95*T, 1, 'first'));
            bw_hz = max(0, f_hi - f_lo);
            features(10) = max(0, min(1, bw_hz / (fs/2)));
        end
        
        features(11) = calculate_spectral_entropy(psd);
        
        psd_norm = psd / (sum(psd) + eps);
        features(12) = calculate_spectral_flatness(psd_norm);
    end
    
    % Category 3: Statistical Analysis (Features 13-16)
    features(13) = mean(real_part);
    
    features(14) = mean(imag_part);
    
    real_power = mean(real_part.^2);
    imag_power = mean(imag_part.^2);
    features(15) = 10*log10((real_power + eps) / (imag_power + eps));
    
    if exist('kurtosis','file')
        features(16) = kurtosis(amplitude) - 3;
    else
        mu = mean(amplitude);
        s2 = var(amplitude) + eps;
        features(16) = mean(((amplitude - mu).^4) ./ (s2^2)) - 3;
    end
    
    % Important checks
    features(~isfinite(features)) = 0;
    if length(features) >= 8
        features(8) = max(0, min(fs/2, features(8)));
    end
    if length(features) >= 9
        features(9) = max(0, min(fs/2, features(9)));
    end
    if length(features) >= 10
        features(10) = max(0, features(10));
    end
    if length(features) >= 11
        features(11) = max(0, min(1, features(11)));
    end
    if length(features) >= 12
        features(12) = max(0, min(1, features(12)));
    end
    
    
    if length(features) ~= 16
        error('Feature vector length mismatch: expected 16, got %d', length(features));
    end
    
    features = reshape(features, 1, 16);
    
    if nargout > 1
        feature_names = {
            'sinr_estimate', 'mean_magnitude', 'std_magnitude', 'peak_to_avg_ratio', 'received_power',...
            'mean_psd_db', 'std_psd_db', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'spectral_entropy', 'spectral_flatness',...
            'mean_real', 'mean_imag', 'iq_power_ratio_db', 'amplitude_kurtosis'
        };
    end
end

% Calculations/helper functions
function centroid = calculate_spectral_centroid(psd, fs)
    if length(psd) < 2 || sum(psd) <= eps
        centroid = 0;
        return;
    end
    
    n_bins = length(psd);
    freq_bins = linspace(0, fs/2, n_bins)';
    psd = psd(:);
    
    total_energy = sum(psd);
    if total_energy > eps
        centroid = sum(freq_bins .* psd) / total_energy;
    else
        centroid = 0;
    end
    
    centroid = max(0, min(fs/2, centroid));
end

function rolloff = calculate_spectral_rolloff(psd, fs, threshold)
    if length(psd) < 2 || sum(psd) <= eps
        rolloff = 0;
        return;
    end
    
    n_bins = length(psd);
    freq_bins = linspace(0, fs/2, n_bins)';
    
    cumulative_energy = cumsum(psd);
    total_energy = cumulative_energy(end);
    
    if total_energy <= eps
        rolloff = 0;
        return;
    end
    
    target_energy = threshold * total_energy;
    rolloff_idx = find(cumulative_energy >= target_energy, 1, 'first');
    
    if isempty(rolloff_idx)
        rolloff = freq_bins(end);
    else
        rolloff = freq_bins(rolloff_idx);
    end
    
    rolloff = max(0, min(fs/2, rolloff));
end

function entropy = calculate_spectral_entropy(psd)
    if length(psd) < 2 || sum(psd) <= eps
        entropy = 0;
        return;
    end
    
    p = psd / sum(psd);
    p(p <= eps) = eps;
    p = p / sum(p);
    
    entropy = -sum(p .* log2(p));
    
    max_entropy = log2(length(psd));
    if max_entropy > 0
        entropy = entropy / max_entropy;
    else
        entropy = 0;
    end
    
    entropy = max(0, min(1, entropy));
end

function flatness = calculate_spectral_flatness(psd_norm)
    if length(psd_norm) < 2 || sum(psd_norm) <= eps
        flatness = 0;
        return;
    end
    
    spectrum_positive = psd_norm(psd_norm > eps);
    if ~isempty(spectrum_positive)
        geometric_mean = exp(mean(log(spectrum_positive)));
        arithmetic_mean = mean(spectrum_positive);
        flatness = geometric_mean / arithmetic_mean;
    else
        flatness = 0;
    end
    
    flatness = max(0, min(1, flatness));
end
