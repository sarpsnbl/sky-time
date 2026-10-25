%% main.m
% =========================================================================
%  Sky Time Estimation — Deep Learning Pipeline
%  Estimates time-of-day from sky images + EXIF DateTime metadata.
%
%  Authors : 220401050 Alkım Gönenç Efe
%            220401067 Damla Parlakyıldız
%            230401114 Sarp Sünbül
%
%  Requirements:
%    • MATLAB Image Processing Toolbox
%    • MATLAB Deep Learning Toolbox
%    • MATLAB Statistics and Machine Learning Toolbox
%    • Pretrained network: resnet18  (run: resnet18 once to auto-download)
%
%  Supported formats : jpg, jpeg, png, dng, heic
%  Input              : images in ./dataset/
%  Output             : per-image predictions + RMSE table
% =========================================================================

clear; clc; close all;
rng(42);

% ── Constants ─────────────────────────────────────────────────────────────
MINUTES_PER_DAY = 1440.0;

% ── Configuration ─────────────────────────────────────────────────────────
cfg.datasetPath    = 'dataset';
cfg.imageFormats   = {'*.jpg','*.jpeg','*.png','*.heic','*.HEIC'};
cfg.inputSize      = [224 224 3];
cfg.kFolds         = 5;
cfg.maxEpochs      = 30;
cfg.miniBatch      = 96;
cfg.learnRate      = 1e-4;
cfg.l2Reg          = 1e-3;
cfg.dateFeatureDim = 4;
cfg.esPatience     = 7;
cfg.featChunkSize  = 100;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║    Sky Time Estimation – MATLAB Pipeline         ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

% ── Step 1 : Load dataset ─────────────────────────────────────────────────
% Scan the ORIGINAL folder to safely extract all the EXIF metadata
cfg.originalPath  = 'dataset'; 
cfg.originalForms = {'*.jpg','*.jpeg','*.png','*.dng','*.heic','*.HEIC'};

fprintf('[1/5]  Scanning "%s" for metadata …\n', cfg.originalPath);

warning('off','all');
[imgPaths, timeLabels, dateFeat] = loadDataset(cfg.originalPath, cfg.originalForms);
warning('on','all');

% --- THE FIX: Reroute the file paths to the resized folder ---
% Now that we have the labels, we swap the paths to point to your new 
% lightning-fast 224x224 JPGs instead of the heavy original files.
for i = 1:numel(imgPaths)
    [~, name, ~] = fileparts(imgPaths{i});
    imgPaths{i}  = fullfile('dataset_224x224', [name, '.jpg']);
end
% -------------------------------------------------------------

N = numel(imgPaths);
fprintf('       %d images loaded and re-routed to 224×224 folder.\n\n', N);

timeMinutes = timeLabels * 60;
angles      = 2 * pi * timeMinutes / MINUTES_PER_DAY;
yEncoded    = [sin(angles), cos(angles)];

% ── Step 2 : Classical features ───────────────────────────────────────────
fprintf('[2/5]  Extracting classical features (chunk size = %d) …\n', cfg.featChunkSize);
classFeat = extractClassicalFeaturesChunked(imgPaths, cfg.inputSize, cfg.featChunkSize);
X_base    = [classFeat, dateFeat];
y_scalar  = timeLabels;
fprintf('       Classical feature matrix: %dx%d\n\n', size(X_base,1), size(X_base,2));
clear classFeat;

% ── Step 3 : K-Fold cross-validation ──────────────────────────────────────
fprintf('[3/5]  %d-Fold Cross-Validation …\n\n', cfg.kFolds);
cv = cvpartition(N, 'KFold', cfg.kFolds);

res_cnn = struct('guesses',[],'actuals',[],'paths',{{}});
res_rf  = struct('guesses',[],'actuals',[],'paths',{{}});
res_svr = struct('guesses',[],'actuals',[],'paths',{{}});

for fold = 1:cfg.kFolds
    fprintf('┌─ Fold %d / %d ─────────────────────────────────\n', fold, cfg.kFolds);

    % Get the master indices for this fold
    trIdxAll = training(cv, fold);
    teIdx    = test(cv, fold);

    % ── 1. ANTI-CHEATING: Split Train into Train & Val ────────────────
    % We isolate the test set completely. Early stopping will now use
    % a 15% holdout strictly from the training data.
    trIndices = find(trIdxAll);
    cvVal = cvpartition(numel(trIndices), 'HoldOut', 0.15);
    actualTrIdx = trIndices(training(cvVal));
    valIdx      = trIndices(test(cvVal));

    % ── Baseline 1 : Random Forest ────────────────────────────────────
    % (Baselines can safely train on the full trIdxAll)
    fprintf('│  [RF]   Training Random Forest (100 trees) …\n');
    rfMdl  = TreeBagger(100, X_base(trIdxAll,:), y_scalar(trIdxAll), ...
                 'Method','regression','MinLeafSize',3);
    rfPred = predict(rfMdl, X_base(teIdx,:));
    res_rf.guesses = [res_rf.guesses;  rfPred];
    res_rf.actuals = [res_rf.actuals;  y_scalar(teIdx)];
    res_rf.paths   = [res_rf.paths;    imgPaths(teIdx)];
    clear rfMdl rfPred;

    % ── Baseline 2 : SVR ──────────────────────────────────────────────
    fprintf('│  [SVR]  Training Support Vector Regressor …\n');
    svrMdl  = fitrsvm(X_base(trIdxAll,:), y_scalar(trIdxAll), ...
                  'KernelFunction','rbf','Standardize',true, ...
                  'KernelScale','auto','BoxConstraint',10);
    svrPred = predict(svrMdl, X_base(teIdx,:));
    res_svr.guesses = [res_svr.guesses;  svrPred];
    res_svr.actuals = [res_svr.actuals;  y_scalar(teIdx)];
    res_svr.paths   = [res_svr.paths;    imgPaths(teIdx)];
    clear svrMdl svrPred;

    % ── Primary : CNN multi-input regression ──────────────────────────
    fprintf('│  [CNN]  Building + training multi-input CNN …\n');
    lgraph  = buildMultiInputCNN(cfg);

    % Build Datastores using the newly split indices
    trainDS = buildDatastore(imgPaths(actualTrIdx), dateFeat(actualTrIdx,:), ...
                             yEncoded(actualTrIdx,:), cfg, true);
    valDS   = buildDatastore(imgPaths(valIdx), dateFeat(valIdx,:), ...
                             yEncoded(valIdx,:), cfg, false);

    outputFn = makeTrainingMonitor(cfg.esPatience, fold, cfg.kFolds);

    opts = trainingOptions('adam', ...
        'MaxEpochs',             cfg.maxEpochs, ...
        'MiniBatchSize',         cfg.miniBatch, ...
        'InitialLearnRate',      cfg.learnRate, ...
        'L2Regularization',      cfg.l2Reg, ...
        'LearnRateSchedule',     'piecewise', ...
        'LearnRateDropFactor',   0.3, ...
        'LearnRateDropPeriod',   15, ...
        'GradientThreshold',     1.0, ...
        'ValidationData',        valDS, ...
        'ValidationFrequency',   20, ...
        'ValidationPatience',    cfg.esPatience, ...
        'OutputFcn',             outputFn, ...
        'Shuffle',               'every-epoch', ...
        'DispatchInBackground',  true, ...
        'Plots',                 'none', ...
        'Verbose',               true, ...
        'VerboseFrequency',      50);

    net = trainNetwork(trainDS, lgraph, opts);

    % ── 2. OOM FIX: Memory-Safe Batched Prediction ────────────────────
    % Replace predictCNN with a batched datastore prediction so we 
    % don't load all test images into RAM at once.
    testDS = buildDatastore(imgPaths(teIdx), dateFeat(teIdx,:), ...
                            yEncoded(teIdx,:), cfg, false);
                            
    % Strip the labels from the test datastore so predict() doesn't error out
    testDS_noLabels = transform(testDS, @(data) data(1:2)); 

    rawPreds = predict(net, testDS_noLabels, 'MiniBatchSize', cfg.miniBatch);

    % Convert the raw [sin, cos] predictions back to minutes
    angles_pred = atan2(rawPreds(:,1), rawPreds(:,2));
    angles_pred(angles_pred < 0) = angles_pred(angles_pred < 0) + 2*pi;
    cnnPredMinutes = (angles_pred / (2*pi)) * MINUTES_PER_DAY;

    res_cnn.guesses = [res_cnn.guesses;  cnnPredMinutes / 60];
    res_cnn.actuals = [res_cnn.actuals;  y_scalar(teIdx)];
    res_cnn.paths   = [res_cnn.paths;    imgPaths(teIdx)];

    % ── 3. OOM FIX: Aggressive Cleanup ────────────────────────────────
    % Close the figure generated by makeTrainingMonitor to free UI RAM
    figName = sprintf('Training Monitor — Fold %d / %d', fold, cfg.kFolds);
    close(findobj('Type', 'Figure', 'Name', figName));

    % Clear massive objects from the workspace
    clear net lgraph trainDS valDS testDS testDS_noLabels outputFn rawPreds angles_pred cnnPredMinutes;
    
    % Force garbage collection and clear GPU
    try
        if gpuDeviceCount > 0
            gpuDevice([]);
            gpuDevice(1);
        end
    catch
    end

    fprintf('└───────────────────────────────────────────────\n\n');
end

% ── Step 4 : Summary ──────────────────────────────────────────────────────
fprintf('[4/5]  Summary\n');
fprintf('═══════════════════════════════════════════════════\n');
printModelResults('CNN (Primary)',          res_cnn);
printModelResults('Random Forest (Base.)', res_rf);
printModelResults('SVR (Baseline)',         res_svr);
fprintf('═══════════════════════════════════════════════════\n\n');

% ── Step 5 : Per-image detail (CNN) ───────────────────────────────────────
fprintf('[5/5]  Per-Image Predictions — CNN Model\n');
fprintf('─────────────────────────────────────────────────────────────\n');
fprintf('%-6s  %-10s  %-10s  %-12s  %s\n', ...
        'Index','Guess','Actual','Abs Err','Image');
fprintf('%s\n', repmat('─',1,80));

allErr = zeros(numel(res_cnn.actuals),1);
for i = 1:numel(res_cnn.actuals)
    g   = res_cnn.guesses(i);
    a   = res_cnn.actuals(i);
    err = circularTimeDiff(g, a) * 60;
    allErr(i) = err;
    [~,fname,ext] = fileparts(res_cnn.paths{i});
    fprintf('%-6d  guess: %-6s  actual: %-6s  %-12s  %s%s\n', ...
            i, hoursToHHMM(g), hoursToHHMM(a), minsToHHMM(err), fname, ext);
end

rmse_min = sqrt(mean(allErr.^2));
fprintf('%s\n', repmat('─',1,80));
fprintf('Overall RMSE: %s  (over all folds)\n\n', minsToHHMM(rmse_min));

T = table((1:numel(res_cnn.actuals))', ...
          arrayfun(@hoursToHHMM, res_cnn.guesses,'UniformOutput',false), ...
          arrayfun(@hoursToHHMM, res_cnn.actuals,'UniformOutput',false), ...
          allErr, res_cnn.paths, ...
          'VariableNames',{'Index','Guess','Actual','AbsErr_min','ImagePath'});
writetable(T, 'sky_time_results.csv');
fprintf('Results saved to sky_time_results.csv\n');


% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function outputFn = makeTrainingMonitor(patience, foldNum, totalFolds)
% MAKETRAININGMONITOR  Returns a single OutputFcn closure that:
%   (1) Draws a live RMSE plot (training in blue, validation in orange).
%   (2) Stops training early when val-RMSE has not improved for `patience`
%       consecutive validation checks.
%
%   The Y-axis shows RMSE in sin/cos space (range 0–√2 ≈ 1.41).
%   The decoded clock-minute RMSE is printed in the summary table.
%
%   A new figure is created per fold; its title shows "Fold X / Y" so
%   you can see all folds' histories side-by-side when running k-fold.

    % ── Plot handles & data buffers ───────────────────────────────────
    hFig   = [];
    hTrain = [];
    hVal   = [];
    hStop  = [];

    iterLog    = [];
    trainLog   = [];
    valIterLog = [];
    valLog     = [];

    % ── Early-stop state ──────────────────────────────────────────────
    bestVal   = Inf;
    esCounter = 0;

    outputFn = @monitor;

    % ------------------------------------------------------------------
    function doStop = monitor(info)
        doStop = false;

        % ── Build figure on first call ────────────────────────────────
        if isempty(hFig) || ~isvalid(hFig)
            [hFig, hTrain, hVal, hStop] = buildPlot(foldNum, totalFolds);
        end

        % ── Log training RMSE (available every iteration) ─────────────
        if ~isnan(info.TrainingRMSE)
            iterLog  = [iterLog,  info.Iteration];    %#ok<AGROW>
            trainLog = [trainLog, info.TrainingRMSE]; %#ok<AGROW>
            set(hTrain, 'XData', iterLog, 'YData', trainLog);
        end

        % ── Log validation RMSE (available every N iterations) ────────
        if ~isempty(info.ValidationRMSE) && ~isnan(info.ValidationRMSE)
            valIterLog = [valIterLog, info.Iteration];       %#ok<AGROW>
            valLog     = [valLog,     info.ValidationRMSE];  %#ok<AGROW>
            set(hVal, 'XData', valIterLog, 'YData', valLog);

            % Auto-scale Y with 10 % headroom
            allY = [trainLog, valLog];
            yLo  = max(0,     min(allY) * 0.90);
            yHi  =            max(allY) * 1.10 + 1e-6;
            ylim(ancestor(hTrain,'axes'), [yLo, yHi]);

            % ── Early-stop logic ──────────────────────────────────────
            if info.ValidationRMSE < bestVal - 1e-4
                bestVal   = info.ValidationRMSE;
                esCounter = 0;
            else
                esCounter = esCounter + 1;
                if esCounter >= patience
                    set(hStop, 'Value', info.Iteration, 'Visible', 'on');
                    drawnow;
                    doStop = true;
                    fprintf('\n  [Early Stop] Val-RMSE flat for %d checks (best = %.5f). Halting.\n', ...
                            patience, bestVal);
                    return;
                end
            end
        end

        drawnow limitrate;   % cap refresh ~20 fps to avoid UI overhead
    end
end

% ── Build the dark-theme figure and return handles ────────────────────────
function [hFig, hTrain, hVal, hStop] = buildPlot(foldNum, totalFolds)

    BG_DARK  = [0.13 0.13 0.16];
    AX_DARK  = [0.10 0.10 0.13];
    TXT_CLR  = [0.88 0.88 0.88];
    GRID_CLR = [0.28 0.28 0.32];

    hFig = figure( ...
        'Name',        sprintf('Training Monitor — Fold %d / %d', foldNum, totalFolds), ...
        'NumberTitle', 'off', ...
        'Color',       BG_DARK, ...
        'Position',    [80 + (foldNum-1)*40, 80 + (foldNum-1)*25, 860, 490]);

    ax = axes(hFig, ...
        'Color',      AX_DARK, ...
        'XColor',     TXT_CLR, ...
        'YColor',     TXT_CLR, ...
        'GridColor',  GRID_CLR, ...
        'GridAlpha',  0.55, ...
        'XGrid',      'on', ...
        'YGrid',      'on', ...
        'FontSize',   11, ...
        'TickDir',    'out', ...
        'Box',        'off');
    hold(ax, 'on');

    % Training curve — cool blue
    hTrain = plot(ax, NaN, NaN, '-', ...
        'Color',       [0.30 0.72 1.00], ...
        'LineWidth',   1.4, ...
        'DisplayName', 'Training RMSE');

    % Validation curve — warm orange with markers
    hVal = plot(ax, NaN, NaN, '-o', ...
        'Color',            [1.00 0.55 0.18], ...
        'LineWidth',        2.2, ...
        'MarkerSize',       5, ...
        'MarkerFaceColor',  [1.00 0.55 0.18], ...
        'DisplayName',      'Validation RMSE');

    % Early-stop vertical line (hidden until triggered)
    hStop = xline(ax, 0, '--', ...
        'Color',                   [0.95 0.25 0.25], ...
        'LineWidth',               1.8, ...
        'Label',                   '  Early Stop', ...
        'LabelVerticalAlignment',  'top', ...
        'LabelHorizontalAlignment','right', ...
        'FontSize',                10, ...
        'FontWeight',              'bold', ...
        'Visible',                 'off');

    xlabel(ax, 'Iteration', 'Color', TXT_CLR, 'FontSize', 12);
    ylabel(ax, 'RMSE  (sin/cos space)', 'Color', TXT_CLR, 'FontSize', 12);
    title(ax, ...
        sprintf('Live Training RMSE — Fold %d / %d', foldNum, totalFolds), ...
        'Color',      [0.95 0.95 0.95], ...
        'FontSize',   14, ...
        'FontWeight', 'bold');

    lgd = legend(ax, 'show', ...
        'TextColor',  TXT_CLR, ...
        'Color',      [0.16 0.16 0.20], ...
        'EdgeColor',  [0.38 0.38 0.42], ...
        'FontSize',   11, ...
        'Location',   'northeast');
end

% ── Chunked classical feature extraction ──────────────────────────────────
function feat = extractClassicalFeaturesChunked(imgPaths, inputSize, chunkSize)
    N    = numel(imgPaths);
    feat = [];
    for start = 1:chunkSize:N
        stop  = min(start + chunkSize - 1, N);
        chunk = extractClassicalFeatures(imgPaths(start:stop), inputSize);
        feat  = [feat; chunk]; %#ok<AGROW>
        clear chunk;
    end
end

% ── Misc helpers ──────────────────────────────────────────────────────────
function diff = circularTimeDiff(a, b)
    d    = mod(abs(a - b), 24);
    diff = min(d, 24 - d);
end

function str = hoursToHHMM(h)
    h  = mod(h, 24);
    hh = floor(h);
    mm = round((h - hh) * 60);
    if mm == 60, hh = hh + 1; mm = 0; end
    str = sprintf('%02d:%02d', mod(hh, 24), mm);
end

function printModelResults(name, res)
    errs = arrayfun(@(g,a) circularTimeDiff(g,a)*60, res.guesses, res.actuals);
    rmse = sqrt(mean(errs.^2));
    mae  = mean(errs);
    fprintf('  %-26s  RMSE: %s   MAE: %s\n', name, minsToHHMM(rmse), minsToHHMM(mae));
end

function str = minsToHHMM(mins)
% Format a duration in minutes as HHhMMm.
    hh  = floor(mins / 60);
    mm  = round(mod(mins, 60));
    if mm == 60, hh = hh + 1; mm = 0; end
    str = sprintf('%02dh%02dm', hh, mm);
end
