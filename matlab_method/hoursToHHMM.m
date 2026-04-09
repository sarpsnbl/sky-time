function s = hoursToHHMM(h)
% HOURSTOHHMM  Convert fractional hours to 'HH:MM' string.
%   Example:  hoursToHHMM(16.583) → '16:35'
    h  = mod(h, 24);
    hh = floor(h);
    mm = round((h - hh) * 60);
    if mm == 60
        hh = hh + 1;
        mm = 0;
    end
    s = sprintf('%02d:%02d', hh, mm);
end
