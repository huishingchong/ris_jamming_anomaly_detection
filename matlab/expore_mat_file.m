% Utility function to explore loaded data interactively
% Lets you load mat file into workspace so you can interact via commandline and 
% lets you look through the matfile structure

function explore_mat_file(filename)
    % To use in MATLAB terminal: explore_mat_file('filename.mat')
    % Save to the workspace then query via variables as usual to explore
    
    if nargin < 1
        [filename, pathname] = uigetfile('*.mat', 'Select MAT file to explore');
        if isequal(filename, 0)
            return;
        end
        filename = fullfile(pathname, filename);
    end
    
    % Load the data
    data = load(filename);
    var_names = fieldnames(data);
    
    fprintf('\nMAT File Explorer\n');
    fprintf('File: %s\n\n', filename);
    
    while true
        fprintf('Available variables:\n');
        for i = 1:length(var_names)
            fprintf(' %d. %s\n', i, var_names{i});
        end
        fprintf('  0. Exit explorer\n');
        
        choice = input('\nSelect variable to examine (number): ');
        
        if isempty(choice) || choice == 0
            break;
        elseif choice > 0 && choice <= length(var_names)
            var_name = var_names{choice};
            var_data = data.(var_name);
            
            fprintf('\n--- Variable: %s ---\n', var_name);
            fprintf('Class: %s\n', class(var_data));
            fprintf('Size: %s\n', mat2str(size(var_data)));
            
            % Display summary based on data type
            if isnumeric(var_data)
                fprintf('Type: Numeric\n');
                if numel(var_data) <= 20
                    fprintf('Values:\n');
                    disp(var_data);
                else
                    fprintf('Min: %g, Max: %g, Mean: %g\n', ...
                           min(var_data(:)), max(var_data(:)), mean(var_data(:)));
                end
            elseif ischar(var_data) || isstring(var_data)
                fprintf('Type: Text\n');
                if length(var_data) <= 200
                    fprintf('Content: %s\n', var_data);
                else
                    fprintf('Content (first 200 chars): %s...\n', var_data(1:200));
                end
            elseif isstruct(var_data)
                fprintf('Type: Structure\n');
                fprintf('Fields: %s\n', strjoin(fieldnames(var_data), ', '));
            elseif iscell(var_data)
                fprintf('Type: Cell Array\n');
                fprintf('Number of cells: %d\n', numel(var_data));
            else
                fprintf('Type: Other (%s)\n', class(var_data));
            end
            
            % Ask if user wants to assign to workspace
            assign_choice = input('\nAssign to workspace? (y/n): ', 's');
            if lower(assign_choice) == 'y'
                assignin('base', var_name, var_data);
                fprintf('Variable %s assigned to base workspace.\n', var_name);
            end
            
        else
            fprintf('Invalid selection.\n');
        end
        
        fprintf('\n' + string(repmat('-', 1, 50)) + '\n');
    end
    
    fprintf('Explorer closed.\n');
end
