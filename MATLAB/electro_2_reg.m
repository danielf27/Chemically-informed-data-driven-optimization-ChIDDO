function vals = electro_2_reg(params, x)

vals = zeros(length(x),1);
for j = 1:length(x)
    vals(j) = electro_2(params, x(j,:));
    if isnan(vals(j))
        vals(j) = 0.01;
    end
    if isinf(vals(j))
        vals(j) = 0.01;
    end

end