function rmse = computeRMSE(guesses, actuals)
% COMPUTERMSE  Root Mean Square Error (circular, hours).
    diffs = circularTimeDiff(guesses, actuals);
    rmse  = sqrt(mean(diffs .^ 2));
end
