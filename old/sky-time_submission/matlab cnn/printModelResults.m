function printModelResults(modelName, res)
% PRINTMODELRESULTS  Print RMSE, MAE and sample predictions for one model.

    g     = res.guesses;
    a     = res.actuals;
    diffs = circularTimeDiff(g, a) * 60;   % convert to minutes

    rmse = sqrt(mean(diffs.^2));
    mae  = mean(diffs);

    fprintf('\n  %-30s\n', modelName);
    fprintf('    RMSE = %6.2f min   |   MAE = %6.2f min\n', rmse, mae);

    nShow = min(3, numel(g));
    for i = 1:nShow
        fprintf('    guess: %s   actual: %s   err: %.1f min\n', ...
                hoursToHHMM(g(i)), hoursToHHMM(a(i)), diffs(i));
    end
end
