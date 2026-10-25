function d = circularTimeDiff(g, a)
% CIRCULARTIMEDITFF  Circular absolute difference on a 24-hour clock.
%   Returns values in [0, 12] hours.
    raw = abs(g - a);
    d   = min(raw, 24 - raw);
end
