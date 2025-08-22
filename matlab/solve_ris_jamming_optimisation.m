%% CORE RIS-BASED JAMMING OPTIMISATION ALGORITHM
% Implements Optimised RIS-based jamming algorithm proposed by 
% Lyu et al. "IRS-Based Wireless Jamming Attacks: When Jammers Can Attack Without Power",
% that was published on IEEE Wireless Communications Letters, vol. 9, no. 10, pp. 1663-1667, Oct. 2020.
% DOI: 10.1109/LWC.2020.3000892
% Implementation is for research purposes

function [beta_opt, theta_opt, sdr_success, bcd_iterations] = solve_ris_jamming_optimisation( ...
    h_r, h_d, G, omega, F, D, use_bcd, max_bcd_iters, bcd_threshold)
    
    N = size(h_r, 2);
    beta_init = 0.5 * ones(N, 1);
    
    [beta_opt, theta_opt, sdr_success, bcd_iterations] = ...
        single_bcd_run(h_r, h_d, G, omega, F, D, beta_init, use_bcd, max_bcd_iters, bcd_threshold);
end

function [beta_final, theta_final, sdr_success, bcd_iterations] = ...
    single_bcd_run(h_r, h_d, G, omega, F, D, beta_init, use_bcd, max_bcd_iters, bcd_threshold)
    % Single BCD run following Lyu et al. algorithm
    
    beta = beta_init;
    theta = zeros(size(h_r, 2), 1);
    
    best_objective = inf;
    best_beta = beta_init;
    best_theta = zeros(size(h_r, 2), 1);
    
    bcd_iterations = 1;
    sdr_success = false;
    prev_obj = inf;
    
    for bcd_iter = 1:max_bcd_iters
        % Step 1: Optimise phase shifts given current amplitudes (Problem P2)
        [theta_new, sdr_worked] = solve_phase_optimisation(h_r, h_d, G, omega, beta, F, D);
        
        % Step 2: Optimise amplitudes given new phase shifts (Problem P3)
        beta_new = solve_amplitude_optimisation(h_r, h_d, G, omega, theta_new);
        
        current_obj = calculate_objective(h_r, h_d, G, omega, beta_new, theta_new);
        
        if current_obj < best_objective
            best_objective = current_obj;
            best_beta = beta_new;
            best_theta = theta_new;
        end
        
        if bcd_iter > 1 && use_bcd
            relative_change = abs(prev_obj - current_obj) / max(abs(prev_obj), 1e-12);
            adaptive_threshold = max(bcd_threshold, 1e-6);
            
            if relative_change < adaptive_threshold
                break;
            end
        end
        
        beta = beta_new;
        theta = theta_new;
        prev_obj = current_obj;
        
        if sdr_worked
            sdr_success = true;
        end
        
        bcd_iterations = bcd_iter;
        
        if ~use_bcd % Terminate if not using bcd
            break;
        end
    end
    
    % Return the best solution found
    beta_final = best_beta;
    theta_final = best_theta;
end

function [theta_opt, sdr_success] = solve_phase_optimisation(h_r, h_d, G, omega, beta, F, D)
    % Solves Problem P2: Phase shift optimisation using SDR and Gaussian randomisation
    
    N = length(beta);
    sdr_success = false;
    
    Gamma = diag(beta);
    alpha = diag(h_r * Gamma) * G * omega;
    psi = h_d * omega; % Direct path contribution
    
    % Check for degenerate case
    if norm(alpha) < 1e-12
        theta_opt = zeros(N, 1);
        return;
    end
    
    % Construct matrix R for SDR formulation (Equation after P2.2 in paper)
    R = [alpha * alpha', alpha * conj(psi);
         psi * alpha', 0];
    
    % Ensure R is hermitian (for numerical stability)
    R = (R + R') / 2;
    
    % Check condition number and apply minimal regularisation only if needed
    cond_R = cond(R);
    if cond_R > 1e12
        reg_factor = 1e-10 * norm(R, 'fro');
        R = R + reg_factor * eye(size(R));
    end
    
    % Solve SDR relaxation (Problem P2.4 in paper)
    try
        cvx_begin quiet
            variable V(N+1, N+1) hermitian semidefinite
            minimize(real(trace(R * V)) + abs(psi)^2)
            subject to
                for n = 1:N
                    V(n, n) == 1;
                end
        cvx_end
        
        if strcmpi(cvx_status, 'Solved') || strcmpi(cvx_status, 'Inaccurate/Solved')
            sdr_success = true;
            
            % Gaussian randomisation (Step 3 in Algorithm 1)
            [U, Sigma, ~] = svd(full(V));
            Sigma = max(real(diag(Sigma)), 0);
            
            best_objective = inf;
            best_theta = zeros(N, 1);
            
            % Generate D random vectors and extract phase shifts
            for d = 1:D
                r = (randn(N+1, 1) + 1j*randn(N+1, 1)) / sqrt(2);
                mu_bar = U * diag(sqrt(Sigma)) * r;
                
                if abs(mu_bar(N+1)) > 1e-10
                    % Extract phase shifts (Equation 6 in paper)
                    theta_bar = -angle(mu_bar(1:N) / mu_bar(N+1));
                    
                    % Quantise to discrete phase set F
                    theta_hat = quantise_phases(theta_bar, F);
                    
                    % Evaluate objective function
                    gamma_d = calculate_objective(h_r, h_d, G, omega, beta, theta_hat);
                    
                    if gamma_d < best_objective
                        best_objective = gamma_d;
                        best_theta = theta_hat;
                    end
                end
            end
            
            theta_opt = best_theta;
            return;
        end        
    catch ME
        fprintf('Phase Optimisation CVX ERROR: %s\n', ME.message);
    end
    
    theta_opt = fallback_phase_optimisation(h_r, h_d, G, omega, beta, F, D);
end

function theta_opt = fallback_phase_optimisation(h_r, h_d, G, omega, beta, F, num_trials)
    fprintf('Fallback used\n');
    N = length(beta);
    Gamma = diag(beta);
    best_objective = inf;
    best_theta = zeros(N, 1);
    
    direct_signal = h_d * omega;
    ris_contributions = diag(h_r * Gamma) .* (G * omega);
    
    for attempt = 1:2
        theta_destructive = zeros(N, 1);
        for n = 1:N
            if abs(ris_contributions(n)) > 1e-12
                if attempt == 1
                    optimal_phase = angle(-direct_signal / ris_contributions(n));
                else
                    optimal_phase = angle(-direct_signal / ris_contributions(n)) + 0.05*randn;
                end
                
                [~, best_idx] = min(abs(F - optimal_phase));
                theta_destructive(n) = F(best_idx);
            end
        end
        
        obj_destructive = calculate_objective(h_r, h_d, G, omega, beta, theta_destructive);
        if obj_destructive < best_objective
            best_objective = obj_destructive;
            best_theta = theta_destructive;
        end
    end
    
    baseline_power = abs(direct_signal)^2;
    if best_objective > 0.5 * baseline_power
        for trial = 1:min(num_trials/8, 15)
            theta_random = F(randi(length(F), N, 1));
            obj_trial = calculate_objective(h_r, h_d, G, omega, beta, theta_random);
            if obj_trial < best_objective
                best_objective = obj_trial;
                best_theta = theta_random;
            end
        end
    end
    
    theta_opt = best_theta;
end

function beta_opt = solve_amplitude_optimisation(h_r, h_d, G, omega, theta)
    % Solve Problem P3: Amplitude optimisation (convex subproblem)
    
    N = length(theta);
    Theta_bar = diag(exp(1j * theta));
    c = diag(h_r) * Theta_bar * G * omega;
    psi = h_d * omega;
    
    % Solve the convex optimisation problem
    try
        cvx_begin quiet
            variable beta_cvx(N, 1)
            minimize(square_abs(beta_cvx' * c + psi))
            subject to
                0 <= beta_cvx <= 1;
        cvx_end
        
        if strcmpi(cvx_status, 'Solved') || strcmpi(cvx_status, 'Inaccurate/Solved')
            beta_opt = beta_cvx;
            return;
        end
    catch ME
        fprintf('Amplitude Optimisation CVX ERROR: %s\n', ME.message);
    end
    
    % Fallback
    if norm(c) > 1e-12
        % Closed-form solution for unconstrained problem
        beta_unconstrained = -real(conj(c) * psi) ./ (abs(c).^2 + 1e-15);
        % Project onto region [0,1]
        beta_opt = max(1e-6, min(1-1e-6, beta_unconstrained));
    else
        beta_opt = 0.5 * ones(N, 1);
    end
end


function obj = calculate_objective(h_r, h_d, G, omega, beta, theta)
    % Calculate objective function value for Problem P1
    % obj = |(h_r^H * Gamma * Theta_bar * G + h_d^H) * omega|^2
    
    Gamma = diag(beta);
    Theta_bar = diag(exp(1j * theta));
    total_channel = h_r * Gamma * Theta_bar * G + h_d;
    obj = abs(total_channel * omega)^2;
end


function theta_q = quantise_phases(theta_cont, F)
    % Quantises continuous phase shifts to discrete set F
    
    % Wrap phases to [0, 2pi) for consistent quantisation  
    theta_mod = mod(theta_cont, 2*pi);
    theta_q = zeros(size(theta_mod));
    
    for i = 1:length(theta_mod)
        % Find closest discrete phase value
        [~, idx] = min(abs(F - theta_mod(i)));
        theta_q(i) = F(idx);
    end
end