function acq_score = acq_calc(X0, mu, sigma, tradeoff, max_val, fun_name, model, X_known, y_known, X_tot)

[test_pred, test_std] = predict(model, X0);
switch fun_name
    case 'EI'
        z_test = -(test_pred - max_val - tradeoff)./test_std;
        z_all = -(mu - max_val - tradeoff)./sigma;
        ave_z = z_test;
        std_z = std(z_all);
        pd = makedist('Normal', 'mu',ave_z, 'sigma',std_z);
        acq_score = -(test_pred - max_val - tradeoff).*normcdf(z_test, ave_z, std_z) + test_std.*pdf(pd,z_test);
    case 'PI'
        z = -(test_pred - max_val - tradeoff)./test_std;
        acq_score = -normcdf(z, test_pred, test_std);
    case 'UCB'
        acq_score = -(test_pred + tradeoff.*test_std);
    case 'MRB'
         b = tradeoff;
        
        combined_X = cat(1,X0, X_known);
        p_dist = pdist(combined_X);
        square = squareform(p_dist);
        [min_val, ind] = min(square(2:end,1));
        
        high_val = max(X_known,[],'all');
        low_val = min(X_known,[],'all');
        grid_max_dist = high_val-low_val;
        DIST = (min_val)/(grid_max_dist);
        FE = test_pred/max(mu);
        STD = test_std/max(sigma);
        similarity = 1/(1+DIST);
        
        if DIST < 0
            DIST = 0;
        end
        acq_score = -(b*(1-similarity) + b*STD + FE);
        
        
end

end