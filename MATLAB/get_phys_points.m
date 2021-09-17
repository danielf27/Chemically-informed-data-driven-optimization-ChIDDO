function [X_all, y_all] = get_phys_points(X_known, y_known, num, params, LB, UB)

dims = size(X_known,2);
num_added = num - length(X_known);
choices = zeros(num_added, dims);
for j = 1:dims
    choices(:,j) = (UB(j)-LB(j)).*rand(num_added,1) + LB(j);
end

y_choices = electro_2_reg(params, choices);

X_all = cat(1,X_known, choices);
y_all = cat(1,y_known, y_choices);

end