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
%    • Pretrained network: squeezenet  (run: squeezenet  once to auto-download)
%
%  Supported formats : jpg, jpeg, png, dng, heic
%  Input              : images in ./dataset/
%  Output             : per-image predictions + RMSE table
% =========================================================================

clear; clc; close all;
rng(42);                  % reproducibility

% ── Configuration ─────────────────────────────────────────────────────────
cfg.datasetPath   = 'dataset';
cfg.imageFormats  = {'*.jpg','*.jpeg','*.png','*.dng','*.heic'};
cfg.inputSize     = [224 224 3];   % squeezenet input
cfg.kFolds        = 5;
cfg.maxEpochs     = 30;
cfg.miniBatch     = 16;
cfg.learnRate     = 1e-4;
cfg.l2Reg         = 1e-4;
cfg.dateFeatureDim = 4;   % [sin(doy), cos(doy), sin(month), cos(month)]

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║    Sky Time Estimation – MATLAB Pipeline         ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

% ── Step 1 : Load dataset ─────────────────────────────────────────────────
fprintf('[1/5]  Scanning "%s" …\n', cfg.datasetPath);
[imgPaths, timeLabels, dateFeat] = loadDataset(cfg.datasetPath, cfg.imageFormats);
N = numel(imgPaths);
fprintf('       %d images loaded with valid DateTime.\n\n', N);

if N < cfg.kFolds
    error('Need at least %d images for %d-fold CV. Found %d.', ...
          cfg.kFolds, cfg.kFolds, N);
end

% ── Step 2 : Classical features (for baseline models) ─────────────────────
fprintf('[2/5]  Extracting classical features …\n');
classFeat = extractClassicalFeatures(imgPaths, cfg.inputSize);
X_base    = [classFeat, dateFeat];     % combined feature matrix
y         = timeLabels;                % time in fractional hours  [0, 24)
fprintf('       Classical feature matrix: %dx%d\n\n', size(X_base,1), size(X_base,2));

% ── Step 3 : K-Fold cross-validation ──────────────────────────────────────
fprintf('[3/5]  %d-Fold Cross-Validation …\n\n', cfg.kFolds);
cv = cvpartition(N, 'KFold', cfg.kFolds);

res_cnn = struct('guesses',[],'actuals',[],'paths',{{}});
res_rf  = struct('guesses',[],'actuals',[],'paths',{{}});
res_svr = struct('guesses',[],'actuals',[],'paths',{{}});

for fold = 1:cfg.kFolds
    fprintf('┌─ Fold %d / %d ─────────────────────────────────\n', fold, cfg.kFolds);

    trIdx = training(cv, fold);
    teIdx = test(cv, fold);

    % ── Baseline 1 : Random Forest ────────────────────────────────────────
    fprintf('│  [RF]   Training Random Forest (100 trees) …\n');
    rfMdl    = TreeBagger(100, X_base(trIdx,:), y(trIdx), ...
                   'Method','regression','MinLeafSize',3);
    rfPred   = predict(rfMdl, X_base(teIdx,:));
    res_rf.guesses = [res_rf.guesses;  rfPred];
    res_rf.actuals = [res_rf.actuals;  y(teIdx)];
    res_rf.paths   = [res_rf.paths;    imgPaths(teIdx)];

    % ── Baseline 2 : SVR ─────────────────────────────────────────────────
    fprintf('│  [SVR]  Training Support Vector Regressor …\n');
    svrMdl   = fitrsvm(X_base(trIdx,:), y(trIdx), ...
                   'KernelFunction','rbf','Standardize',true, ...
                   'KernelScale','auto','BoxConstraint',10);
    svrPred  = predict(svrMdl, X_base(teIdx,:));
    res_svr.guesses = [res_svr.guesses;  svrPred];
    res_svr.actuals = [res_svr.actuals;  y(teIdx)];
    res_svr.paths   = [res_svr.paths;    imgPaths(teIdx)];

    % ── Primary : CNN multi-input regression ─────────────────────────────
    fprintf('│  [CNN]  Building + training multi-input CNN …\n');
    lgraph   = buildMultiInputCNN(cfg);
    trainDS  = buildDatastore(imgPaths(trIdx), dateFeat(trIdx,:), ...
                              y(trIdx), cfg, true);
    valDS    = buildDatastore(imgPaths(teIdx), dateFeat(teIdx,:), ...
                              y(teIdx),  cfg, false);

    opts = trainingOptions('adam', ...
        'MaxEpochs',         cfg.maxEpochs, ...
        'MiniBatchSize',     cfg.miniBatch, ...
        'InitialLearnRate',  cfg.learnRate, ...
        'L2Regularization',  cfg.l2Reg, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'ValidationData',    valDS, ...
        'ValidationFrequency', 10, ...
        'Shuffle',           'every-epoch', ...
        'Plots',             'none', ...
        'Verbose',           false);

    net      = trainNetwork(trainDS, lgraph, opts);
    cnnPred  = predictCNN(net, imgPaths(teIdx), dateFeat(teIdx,:), cfg);

    res_cnn.guesses = [res_cnn.guesses;  cnnPred];
    res_cnn.actuals = [res_cnn.actuals;  y(teIdx)];
    res_cnn.paths   = [res_cnn.paths;    imgPaths(teIdx)];

    fprintf('└───────────────────────────────────────────────\n\n');
end

% ── Step 4 : Print summary results ────────────────────────────────────────
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
        'Index','Guess','Actual','Abs Err(min)','Image');
fprintf('%s\n', repmat('─',1,80));

allErr = zeros(numel(res_cnn.actuals),1);
for i = 1:numel(res_cnn.actuals)
    g   = res_cnn.guesses(i);
    a   = res_cnn.actuals(i);
    err = circularTimeDiff(g, a) * 60;   % minutes
    allErr(i) = err;
    [~,fname,ext] = fileparts(res_cnn.paths{i});
    fprintf('%-6d  guess: %-6s  actual: %-6s  %7.1f min     %s%s\n', ...
            i, hoursToHHMM(g), hoursToHHMM(a), err, fname, ext);
end

rmse_min = sqrt(mean(allErr.^2));
fprintf('%s\n', repmat('─',1,80));
fprintf('error margin: %.2f min (RMSE over all folds)\n\n', rmse_min);

% ── Optional: Save results to CSV ─────────────────────────────────────────
T = table((1:numel(res_cnn.actuals))', ...
          arrayfun(@hoursToHHMM, res_cnn.guesses,'UniformOutput',false), ...
          arrayfun(@hoursToHHMM, res_cnn.actuals,'UniformOutput',false), ...
          allErr, res_cnn.paths, ...
          'VariableNames',{'Index','Guess','Actual','AbsErr_min','ImagePath'});
writetable(T, 'sky_time_results.csv');
fprintf('Results saved to sky_time_results.csv\n');
