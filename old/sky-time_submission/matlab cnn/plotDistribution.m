%% plotDistributionBias.m
% =========================================================================
%  Compares the distribution of ACTUAL photo times vs GUESSED times to
%  reveal whether the model simply learned the dataset's time distribution.
%
%  Reads : sky_time_results.csv  (columns: Index, Guess, Actual, AbsErr_min, ImagePath)
%  Output: figure + distribution_bias.png
% =========================================================================

clear; clc; close all;

% ── Load results ──────────────────────────────────────────────────────────
T = readtable('sky_time_results.csv');

% Parse HH:MM strings → fractional hours
actualHr = parseHHMM(T.Actual);
guessHr  = parseHHMM(T.Guess);

% ── Bin into 30-minute slots (48 bins over 24 h) ──────────────────────────
binWidth = 0.5;                          % hours (30 min)
edges    = 0 : binWidth : 24;            % 49 edges → 48 bins
binCtrs  = edges(1:end-1) + binWidth/2;  % bin centres for labelling

actualCounts = histcounts(actualHr, edges);
guessCounts  = histcounts(guessHr,  edges);

% ── Plot ──────────────────────────────────────────────────────────────────
BG    = [0.10 0.10 0.13];
AX    = [0.14 0.14 0.18];
CGRID = [0.28 0.28 0.34];
CACT  = [0.30 0.72 1.00];   % blue  — dataset (actual)
CGUESS= [1.00 0.55 0.18];   % orange — model guesses
CTXT  = [0.90 0.90 0.90];

fig = figure('Name','Distribution Bias Check', ...
             'NumberTitle','off', ...
             'Color', BG, ...
             'Position',[100 100 1100 560]);

ax = axes(fig, ...
    'Color',      AX, ...
    'XColor',     CTXT, ...
    'YColor',     CTXT, ...
    'GridColor',  CGRID, ...
    'GridAlpha',  0.5, ...
    'XGrid',      'on', ...
    'YGrid',      'on', ...
    'FontSize',   11, ...
    'TickDir',    'out', ...
    'Box',        'off');
hold(ax, 'on');

% Draw bars — actual behind, guess in front with slight transparency
bAct   = bar(ax, binCtrs, actualCounts, ...
    'BarWidth',   1.0, ...
    'FaceColor',  CACT, ...
    'EdgeColor',  'none', ...
    'FaceAlpha',  0.85, ...
    'DisplayName','Dataset (actual times)');

bGuess = bar(ax, binCtrs, guessCounts, ...
    'BarWidth',   1.0, ...
    'FaceColor',  CGUESS, ...
    'EdgeColor',  'none', ...
    'FaceAlpha',  0.55, ...
    'DisplayName','Model guesses');

% ── Axis labels & ticks ───────────────────────────────────────────────────
% Show HH:00 labels every 2 hours
tickHrs = 0:2:24;
xticks(ax, tickHrs);
xticklabels(ax, arrayfun(@(h) sprintf('%02d:00',h), tickHrs, 'UniformOutput',false));
xtickangle(ax, 45);

xlim(ax, [0 24]);
xlabel(ax, 'Time of Day',  'Color', CTXT, 'FontSize', 13);
ylabel(ax, 'Photo Count',  'Color', CTXT, 'FontSize', 13);

title(ax, 'Dataset Distribution vs. Model Guess Distribution (30-min bins)', ...
    'Color',      [0.96 0.96 0.96], ...
    'FontSize',   15, ...
    'FontWeight', 'bold');

% ── KDE overlay (smoothed curves) ─────────────────────────────────────────
% Gives a cleaner view of the shape alignment
fineX    = linspace(0, 24, 600);
bwHr     = 0.6;   % bandwidth in hours

kdeAct   = kernelDensity(actualHr, fineX, bwHr) * numel(actualHr) * binWidth;
kdeGuess = kernelDensity(guessHr,  fineX, bwHr) * numel(guessHr)  * binWidth;

plot(ax, fineX, kdeAct,   '-',  'Color', CACT   * 0.85, 'LineWidth', 2.2, ...
     'HandleVisibility','off');
plot(ax, fineX, kdeGuess, '--', 'Color', CGUESS * 0.85, 'LineWidth', 2.2, ...
     'HandleVisibility','off');

% ── Legend ────────────────────────────────────────────────────────────────
lgd = legend(ax, 'show', ...
    'TextColor',  CTXT, ...
    'Color',      [0.16 0.16 0.20], ...
    'EdgeColor',  [0.38 0.38 0.44], ...
    'FontSize',   12, ...
    'Location',   'northwest');

% ── Annotation: correlation between distributions ─────────────────────────
r   = corr(actualCounts', guessCounts');
kld = klDivergence(actualCounts, guessCounts);

annStr = sprintf('Distribution correlation r = %.3f\nKL divergence = %.3f', r, kld);
annotation(fig, 'textbox', [0.72 0.78 0.22 0.12], ...
    'String',    annStr, ...
    'Color',     CTXT, ...
    'BackgroundColor', [0.16 0.16 0.20], ...
    'EdgeColor', [0.38 0.38 0.44], ...
    'FontSize',  11, ...
    'LineWidth', 1.2, ...
    'FitBoxToText','on');

drawnow;

% ── Save figure ───────────────────────────────────────────────────────────
outFile = 'distribution_bias.png';
exportgraphics(fig, outFile, 'Resolution', 180, 'BackgroundColor', BG);
fprintf('Figure saved to %s\n', outFile);

% ── Console summary ───────────────────────────────────────────────────────
fprintf('\n── Distribution Bias Check ──────────────────────────\n');
fprintf('  Total images        : %d\n',   numel(actualHr));
fprintf('  Bin width           : %.0f min\n', binWidth * 60);
fprintf('  Distribution r      : %.4f\n', r);
fprintf('  KL divergence       : %.4f\n', kld);
fprintf('  (r → 1 and KL → 0 means the model is echoing the dataset curve)\n');

% ── Top 5 most over-guessed bins ─────────────────────────────────────────
diff = guessCounts - actualCounts;
[sorted, idx] = sort(diff, 'descend');
fprintf('\n  5 most over-guessed time slots:\n');
for k = 1:5
    i = idx(k);
    hStart = edges(i);
    hEnd   = edges(i+1);
    fprintf('    %02d:%02d–%02d:%02d   guesses: %d   actuals: %d   delta: %+d\n', ...
        floor(hStart), round(mod(hStart,1)*60), ...
        floor(hEnd),   round(mod(hEnd,1)*60), ...
        guessCounts(i), actualCounts(i), sorted(k));
end

fprintf('\n  5 most under-guessed time slots:\n');
for k = numel(idx):-1:numel(idx)-4
    i = idx(k);
    hStart = edges(i);
    hEnd   = edges(i+1);
    fprintf('    %02d:%02d–%02d:%02d   guesses: %d   actuals: %d   delta: %+d\n', ...
        floor(hStart), round(mod(hStart,1)*60), ...
        floor(hEnd),   round(mod(hEnd,1)*60), ...
        guessCounts(i), actualCounts(i), diff(i));
end

fprintf('─────────────────────────────────────────────────────\n\n');


% =========================================================================
%  LOCAL HELPERS
% =========================================================================

function hr = parseHHMM(cellArr)
% Convert a cell array of 'HH:MM' strings to a numeric vector of
% fractional hours.  Handles both cell and string arrays.
    if isstring(cellArr)
        cellArr = cellstr(cellArr);
    end
    n  = numel(cellArr);
    hr = zeros(n, 1);
    for i = 1:n
        parts  = strsplit(strtrim(cellArr{i}), ':');
        hr(i)  = str2double(parts{1}) + str2double(parts{2}) / 60;
    end
end

function density = kernelDensity(data, xq, bw)
% Gaussian kernel density estimate at query points xq.
%   bw – bandwidth in same units as data
    n       = numel(data);
    density = zeros(size(xq));
    for j = 1:n
        density = density + exp(-0.5 * ((xq - data(j)) / bw).^2);
    end
    density = density / (n * bw * sqrt(2*pi));
end

function kld = klDivergence(p, q)
% Symmetrised KL divergence between two count distributions.
%   Small epsilon prevents log(0).
    eps = 1e-9;
    p   = (p + eps) / sum(p + eps);
    q   = (q + eps) / sum(q + eps);
    kld = 0.5 * sum(p .* log(p ./ q)) + 0.5 * sum(q .* log(q ./ p));
end