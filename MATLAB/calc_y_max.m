function y_known_max = calc_y_max(y_known)
y_known_max = zeros(length(y_known),1);

for val = 1:length(y_known)
    if val == 1
        y_known_max(val) = y_known(val);
    else
        if y_known(val) > y_known_max(val-1)
            y_known_max(val) = y_known(val);
        else
            y_known_max(val) = y_known_max(val-1);
        end
    end
end

end