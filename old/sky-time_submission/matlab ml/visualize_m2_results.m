%% visualize_m2_results.m
%  Reads the per-fold CSV files written by main.m (fold_1_predictions.csv …
%  fold_5_predictions.csv) and produces two publication figures:
%
%    m2_fold_mae.png   — grouped bar chart: MAE per fold for each model
%                        (reads from fold_N_predictions.csv; ensemble only
%                         is available per-fold, so we plot that here plus
%                         the overall model summary from Step 4 if the
%                         caller passes results in; otherwise ensemble only)
%
%    m2_scatter.png    — predicted vs actual scatter coloured by error,
%                        collated across all folds from fold_N_predictions.csv
%
%  USAGE (after running main.m):
%    visualize_m2_results          % reads fold_1..5_predictions.csv

% ─────────────────────────────────────────────────────────────────────────
%  1.  Load all fold prediction files
% ─────────────────────────────────────────────────────────────────────────
nFolds = 5;
allGuess  = [];
allActual = [];
foldMAE   = nan(nFolds, 4); % Cols: SVR, GBT, RF, Ensemble

for f = 1:nFolds
    fname = sprintf('fold_%d_predictions.csv', f);
    if ~isfile(fname)
        warning('File not found: %s — skipping fold %d.', fname, f);
        continue
    end
    T = readtable(fname);
    
    % Use Ens_Guess for the collective scatter plot
    guessH  = hhmm2hours(T.Ens_Guess);
    actualH = hhmm2hours(T.Actual);
    
    allGuess  = [allGuess;  guessH];   %#ok<AGROW>
    allActual = [allActual; actualH];  %#ok<AGROW>
    
    foldMAE(f, :) = [mean(T.SVR_Err), mean(T.GBT_Err), mean(T.RF_Err), mean(T.Ens_Err)];
end

% ─────────────────────────────────────────────────────────────────────────
%  2.  Figure 1 — per-fold MAE Grouped Bar Chart (All Models)
% ─────────────────────────────────────────────────────────────────────────
fig1 = figure('Color','white','Position', [100 100 800 450]);

validIdx = any(~isnan(foldMAE), 2);
validFolds = find(validIdx);
data = foldMAE(validIdx, :);

% Premium Light Mode colors: Soft Palette
colors = [
    0.34, 0.70, 0.50;  % SVR: Soft Green
    0.95, 0.60, 0.20;  % GBT: Soft Orange
    0.20, 0.60, 0.86;  % RF: Soft Blue
    0.60, 0.40, 0.80   % Ensemble: Soft Purple
];

b = bar(validFolds, data, 'grouped', 'EdgeColor', 'none', 'FaceAlpha', 0.85);
for k = 1:4
    b(k).FaceColor = colors(k, :);
end

hold on;

% Mean line for Ensemble
ensMean = mean(data(:, 4));
yline(ensMean, '--', sprintf('Ensemble Mean: %.1f min', ensMean), ...
      'Color', [0.3 0.3 0.3], 'LineWidth', 1.2, ...
      'LabelHorizontalAlignment', 'right', 'FontSize', 10, 'FontName', 'Outfit');

ax1 = gca;
ax1.XTick = validFolds;
ax1.XTickLabel = arrayfun(@(f) sprintf('Fold %d', f), validFolds, 'UniformOutput', false);
ax1.YLabel.String = 'MAE (minutes)';
ax1.FontSize  = 11;
ax1.FontName  = 'Outfit';
ax1.Box       = 'off';
ax1.GridAlpha = 0.1;
ax1.YGrid     = 'on';
ax1.TickDir   = 'out';

title('Method 2 — Model Comparison: Per-Fold Validation MAE', ...
      'FontSize', 15, 'FontWeight', 'bold', 'FontName', 'Outfit', 'Color', [0.1 0.1 0.1]);

legend({'SVR', 'GBT', 'RF', 'Ensemble'}, 'Location', 'northeastoutside', ...
       'Box', 'off', 'FontName', 'Outfit', 'FontSize', 10);
hold off;

outFoldMAE = fullfile(fileparts(mfilename('fullpath')), 'm2_fold_mae.png');
exportgraphics(fig1, outFoldMAE, 'Resolution', 200);
fprintf('Saved:  %s\n', outFoldMAE);

% ─────────────────────────────────────────────────────────────────────────
%  3.  Figure 2 — Predicted vs Actual scatter (ensemble, all folds)
% ─────────────────────────────────────────────────────────────────────────
if isempty(allGuess)
    warning('No prediction data loaded; skipping scatter plot.');
    return
end

errMin = cyclicDiffHours(allGuess, allActual) * 60;

fig2 = figure('Color', 'white', 'Position', [150 150 650 600]);

% ±30 min band - Softer green
x_diag = 0:0.1:24;
fill([x_diag, fliplr(x_diag)], ...
     [min(x_diag+0.5, 24), fliplr(max(x_diag-0.5, 0))], ...
     [0.18, 0.80, 0.44], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % #2ecc71
hold on;

% Perfect prediction diagonal - Subtle gray
plot([0 24], [0 24], 'Color', [0.4 0.4 0.4], 'LineWidth', 1.5, 'LineStyle', '-');

% Scatter coloured by error - Premium Gradient (Blue to Gold)
sc = scatter(allActual, allGuess, 25, errMin, 'filled', ...
             'MarkerFaceAlpha', 0.5, 'MarkerEdgeColor', 'none');

% Premium Colormap: Deep Blue to Vibrant Gold
nC = 256;
cmap = [linspace(0.1, 0.95, nC)', linspace(0.3, 0.85, nC)', linspace(0.6, 0.1, nC)'];
colormap(fig2, cmap);

cbar = colorbar;
cbar.Label.String = 'Absolute Error (min)';
cbar.Label.FontSize = 10;
cbar.Label.FontName = 'Outfit';
cbar.FontSize = 9;
clim([0, min(max(errMin), 120)]); % Cap at 120 for visual contrast

ax2 = gca;
ax2.XLabel.String = 'Actual Time (hours)';
ax2.YLabel.String = 'Predicted Time (hours)';
ax2.FontSize  = 11;
ax2.FontName  = 'Outfit';
ax2.Box       = 'off';
ax2.XLim      = [0 24];
ax2.YLim      = [0 24];
ax2.XTick     = 0:3:24;
ax2.YTick     = 0:3:24;
ax2.XTickLabel = arrayfun(@(h) sprintf('%02d:00',h), 0:3:24, 'UniformOutput',false);
ax2.YTickLabel = arrayfun(@(h) sprintf('%02d:00',h), 0:3:24, 'UniformOutput',false);
ax2.GridAlpha = 0.1;
ax2.XGrid = 'on'; ax2.YGrid = 'on';
ax2.TickDir = 'out';
ax2.Color = 'white';

title(sprintf('Method 2 Ensemble — Predicted vs Actual (MAE = %.1f min)', mean(errMin)), ...
      'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Outfit', 'Color', [0.1 0.1 0.1]);

% Legend
lgd = legend({'±30 min band', 'Perfect Match', 'Predictions'}, ...
             'Location', 'northwest', 'FontSize', 10, 'Box', 'off', 'FontName', 'Outfit');
hold off;

outScatter = fullfile(fileparts(mfilename('fullpath')), 'm2_scatter.png');
exportgraphics(fig2, outScatter, 'Resolution', 200);
fprintf('Saved:  %s\n', outScatter);

% ─────────────────────────────────────────────────────────────────────────
%  LOCAL HELPERS
% ─────────────────────────────────────────────────────────────────────────
function h = hhmm2hours(cellStr)
%HHMM2HOURS  Convert a cell array of 'HH:MM' strings to decimal hours.
    h = zeros(numel(cellStr),1);
    for k = 1:numel(cellStr)
        s  = char(cellStr{k});
        parts = strsplit(s, ':');
        h(k) = str2double(parts{1}) + str2double(parts{2})/60;
    end
end

function d = cyclicDiffHours(g, a)
%CYCLICDIFFHOURS  Shortest circular distance between two time vectors (hours).
    d = abs(g - a);
    d = min(d, 24 - d);
end
