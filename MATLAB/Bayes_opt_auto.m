%% Bayesian Optimization for experiments (MATLAB)

%% Define hyperparameters
% If any of the dimensions have a different range, input LB and UB as
% below:
% LB = [0.1 0.1 0.1];
% UB = [3 2 2];
% If all the dimensions have the same range, use single values as below:
LB = [100, 100];
UB = [1000, 1000];

% Select how much noise (if any) to add to the simulated experimental
% values. The string in 'noise_str' will be used when naming files.
noise = 0.025;
noise_str = '25';

date = '2021-06-01';

% dims = # of variable dimensions
dims = 2;

% init_num = # of initial data points to use
init_num = 10;

% batch_size = # of new experiments per batch of BO
batch_size = 3;

% total_num = total # of experiments to run before stopping
total_num = 50;

% num = number of points in each dimension.
num = 20;

% Choose from MRB, PI, UCB, EI
acq_fun = 'MRB';

% Choose from BO, ChIDDO
BO_ChIDDO = 'ChIDDO';


%% Create comparison grid of design space
x_1 = linspace(LB(1),UB(1),num);
x_2 = linspace(LB(2),UB(2),num);
X_tot = zeros(num*num,dims);

for j = 1:length(x_1)
    for k = 1:length(x_2)
    % If adding another dimension, n would be:
%         n = num*num*(j-1) + num*(k-1) + m;
        n = num*(j-1) + k;
        X_tot(n, :) = [x_1(j) x_2(k)];
    end
end

%% Define parameters/parameter ranges and create a set of alternative parameters
% params = the physics model parameters that is your estimated parameters
% of the system
params = [0.95, 1, 5e-8, 5e-10, 0.5, 0.5];
% param_std = estimated standard deviation of the possible parameter
% values. This is used to create alternate models for testing purposes.
param_std = [0.05, 0.05, 2e-8, 2e-10, 0.15, 0.15];

% Generate alternate parameter sets for testing purposes. 
alt_num = 20;
alt_params = zeros(alt_num,length(params));
for j = 1:length(params)
    alt_params(:,j) = normrnd(params(j), param_std(j), [alt_num,1]);   
end
alts_filename = ['alts_' date '.csv'];
writematrix(alt_params, alts_filename);

% alt_name = ['alts_' date '.csv'];
% alt_table = readtable(alt_name, 'ReadVariableNames', false);
% alt_params = table2array(alt_table);

%% Calculate grid of points for all of the alternate models
% Calculate data for each of the alternate models
data = zeros(length(X_tot),alt_num);
for alt = 1:alt_num
    for row = 1:length(X_tot)
        data(row, alt) = electro_2(alt_params(alt,:), X_tot(row,:)) + 2*noise*rand - noise;
        if data(row, alt) < 0
            data(row, alt) = 0.01;
        elseif data(row, alt) > 1
                data(row,alt) = 0.99;
        end
    end
end

%% Calculate the max value and location for each of alternate models
% Calculate max value for each of the alt models and save as csv
[alt_max, alt_inds] = max(data);
alt_max_data = cat(2,X_tot(alt_inds,:),alt_max.');
alt_filename = ['alt_max_data_' date '.csv'];
writematrix(alt_max_data, alt_filename);

%% Select initial experiments
if length(UB)==1
    init_points = (UB-LB).*rand(init_num,dims) + LB;
    % If you're variable only goes to a certain decimal point, use the
    % round function below:
%     init_points = round(init_points, 1);
else
    init_points = zeros(init_num, dims);
    for j = 1:length(UB)
        init_points(:,j) = (UB(j)-LB(j)).*rand(init_num,1) + LB(j);
    end
end

%% Save the initial points in a file
init_filename = ['Init_points_electro_2_', date, '.csv'];
writematrix(init_points, init_filename);

%% Run BO/ChIDDO

% Optimize each of the alternative models
for alt = 1:alt_num
    alt
    X_known = init_points;
    y_known = electro_2_reg(alt_params(alt,:), X_known) + 2*noise*rand - noise;
    y_known(y_known < 0) = 0.01;
    y_known(y_known > 1) = 0.99;
    % tot_batches = total # of batches before total_num is reached
    tot_batches = floor((total_num - init_num)/batch_size);

% init_tradeoff = value that corresponds to exploration vs. exploitation.
% Decreases in value as batches increase
    if strcmp(acq_fun,'MRB')
        tradeoff = linspace(1,0,tot_batches);
    elseif strcmp(acq_fun,'UCB')
        tradeoff = linspace(4,0,tot_batches);
    else
        tradeoff = logspace(-1.3,-7,tot_batches);
    end
        
    phys_params = params;
    
    for batch = 1:tot_batches
        batch
        if strcmp(BO_ChIDDO,'ChIDDO')
            [X_used, y_used] = get_phys_points(X_known, y_known, total_num, phys_params, LB, UB);
        end
        
        new_tradeoff = tradeoff(batch);
        
        if strcmp(BO_ChIDDO, 'BO')
            [X_new, preds, stds] = run_learner(X_tot, X_known, y_known, batch_size, new_tradeoff, LB, UB, acq_fun);
        else
            [X_new, preds, stds] = run_learner(X_tot, X_used, y_used, batch_size, new_tradeoff, LB, UB, acq_fun);
        end
        
        X_known = cat(1,X_known,X_new);
        y_new = electro_2_reg(alt_params(alt,:), X_new) + 2*noise*rand - noise;
        y_known = cat(1,y_known,y_new);
        y_known(y_known < 0) = 0.01;
        y_known(y_known > 1) = 0.99;
        
        known_data = cat(2,X_known,y_known);
        
        if strcmp(BO_ChIDDO, 'ChIDDO')

            all_data = cat(2,X_known,y_known);
            all_data_tbl = array2table(all_data);
            all_data_tbl.Properties.VariableNames = {'Conc_A' 'Conc_B' 'FE'};
            
            all_data_tbl.Conc_A(isnan(all_data_tbl.Conc_A)) = 500;
            all_data_tbl.Conc_B(isnan(all_data_tbl.Conc_B)) = 500;
            all_data_tbl.FE(isnan(all_data_tbl.FE)) = 0.01;
            all_data_tbl.Conc_A(isinf(all_data_tbl.Conc_A)) = 500;
            all_data_tbl.Conc_B(isinf(all_data_tbl.Conc_B)) = 500;
            all_data_tbl.FE(isinf(all_data_tbl.FE)) = 0.01;
            options = statset('MaxIter', 500);
            mdl = fitnlm(all_data_tbl, @(b,x)electro_2_reg(b,x), params, 'Options',options);
            params_table = mdl.Coefficients.Estimate;
            phys_params = params_table.';
        end
        
        if batch == tot_batches
            known_filename = ['known_points' noise_str '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(alt) '_' date '.csv'];
            writematrix(known_data, known_filename);
        end
        
        preds_filename = ['preds' noise_str '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(alt) '_batch_' num2str(batch) '_' date '.csv'];
        writematrix(preds, preds_filename);

        stds_filename = ['std' noise_str '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(alt) '_batch_' num2str(batch) '_' date '.csv'];
        writematrix(stds, stds_filename);
    end
end

%% Plot data and save error percent and distance files

for j = 1:alt_num
    known_name = ['known_points' noise_str '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(j) '_' date '.csv'];
    known_table = readtable(known_name, 'ReadVariableNames', false);
    known_data = table2array(known_table);
    

    preds = zeros(length(X_tot), tot_batches);
    for k = 1:tot_batches
        pred_name = ['preds' noise_str '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(j) '_batch_' num2str(k) '_' date '.csv'];
        pred_table = readtable(pred_name, 'ReadVariableNames', false);
        preds(:,k) = table2array(pred_table);
    end
   
    
end
    
%% Plot error percent

for j = 1:alt_num
    known_name = ['known_points' noise '_' acq_fun '_' BO_ChIDDO '_alt_' num2str(j) '_' date '.csv'];
    known_table = readtable(known_name, 'ReadVariableNames', false);
    known_data = table2array(known_table);
    
    y_max_vals = calc_y_max(known_data(:,3));
    
    figure();
    plot(1:length(y_max_vals),y_max_vals, 'LineWidth', 3)
    
end