%% main.m  —  Method 2
% =========================================================================
%  Sky Time Estimation — Statistical Pipeline (SVR + GBT + RF Ensemble)
%  Estimates time-of-day from sky images using hand-crafted features only.
%
%  Authors : 220401050 Alkım Gönenç Efe
%            220401067 Damla Parlakyıldız
%            230401114 Sarp Sünbül
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
cfg.featChunkSize  = 100;

fprintf('╔══════════════════════════════════════════════════╗\n');
fprintf('║  Sky Time Estimation – Statistical Pipeline     ║\n');
fprintf('║  Method 2 : Ensemble (SVR + GBT + RF)           ║\n');
fprintf('╚══════════════════════════════════════════════════╝\n\n');

% ── Step 0 : Build blacklist from audit.csv ───────────────────────────────
fprintf('[0/5]  Loading audit.csv to blacklist 100 worst images …\n');
if exist('audit.csv', 'file')
    auditT = readtable('audit.csv');
    auditT = sortrows(auditT, 'error_min', 'descend');
    nBlack = min(100, height(auditT));
    worst100 = auditT(1:nBlack, :);
    blacklistStems = cellfun(@fileStem, worst100.path, 'UniformOutput', false);
    fprintf('       Blacklisted %d image stems.\n\n', nBlack);
else
    blacklistStems = {};
    fprintf('       audit.csv not found, skipping blacklist.\n\n');
end

% ── Step 1 : Load dataset ─────────────────────────────────────────────────
fprintf('[1/5]  Scanning dataset for metadata …\n');
warning('off','all');
[imgPaths, timeLabels, dateFeat] = loadDataset(cfg.datasetPath, cfg.imageFormats);
warning('on','all');

% Re-route paths to the 224×224 folder
for i = 1:numel(imgPaths)
    [~, name, ~] = fileparts(imgPaths{i});
    imgPaths{i}  = fullfile('dataset_224x224', [name, '.jpg']);
end

% ── Remove blacklisted images ─────────────────────────────────────────────
imgStems = cellfun(@fileStem, imgPaths, 'UniformOutput', false);
keepMask = ~ismember(imgStems, blacklistStems);
imgPaths   = imgPaths(keepMask);
timeLabels = timeLabels(keepMask);
dateFeat   = dateFeat(keepMask, :);
N = numel(imgPaths);
fprintf('       %d images loaded.\n\n', N);

% ── Circular encoding of target ───────────────────────────────────────────
timeMinutes = timeLabels * 60;
angles      = 2 * pi * timeMinutes / MINUTES_PER_DAY;
ySin        = sin(angles);
yCos        = cos(angles);

% ── Step 2 : Classical features ───────────────────────────────────────────
fprintf('[2/5]  Extracting classical features …\n');
classFeat = extractClassicalFeaturesChunked(imgPaths, cfg.inputSize, cfg.featChunkSize);
X_all     = [classFeat, dateFeat];
y_scalar  = timeLabels;
fprintf('       Feature matrix: %d × %d\n\n', size(X_all,1), size(X_all,2));
clear classFeat;

% ── Step 3 : K-Fold cross-validation ──────────────────────────────────────
fprintf('[3/5]  %d-Fold Cross-Validation …\n\n', cfg.kFolds);
cv = cvpartition(N, 'KFold', cfg.kFolds);

% Result accumulators
res_svr   = struct('guesses',[],'actuals',[],'paths',{{}});
res_boost = struct('guesses',[],'actuals',[],'paths',{{}});
res_rf    = struct('guesses',[],'actuals',[],'paths',{{}});
res_ens   = struct('guesses',[],'actuals',[],'paths',{{}});

for fold = 1:cfg.kFolds
    fprintf('┌─ Fold %d / %d ─────────────────────────────────\n', fold, cfg.kFolds);

    trIdx = training(cv, fold);
    teIdx = test(cv, fold);

    X_tr = X_all(trIdx,:);    X_te = X_all(teIdx,:);
    ySin_tr = ySin(trIdx);    yCos_tr = yCos(trIdx);
    
    % ── Standardise features ──────────────────────────────────────────────
    [X_tr_z, mu, sigma] = zscore(X_tr);
    sigma(sigma == 0) = 1;
    X_te_z = (X_te - mu) ./ sigma;

    % ── Model 1 : SVR (Sin/Cos) ───────────────────────────────────────────
    fprintf('│  [SVR]  Training …\n');
    svrSin = fitrsvm(X_tr_z, ySin_tr, 'KernelFunction','gaussian','KernelScale','auto','Standardize',false);
    svrCos = fitrsvm(X_tr_z, yCos_tr, 'KernelFunction','gaussian','KernelScale','auto','Standardize',false);
    
    pSinS = predict(svrSin, X_te_z); pCosS = predict(svrCos, X_te_z);
    angS  = atan2(pSinS, pCosS); angS(angS < 0) = angS(angS < 0) + 2*pi;
    res_svr.guesses = [res_svr.guesses; (angS/(2*pi))*24];
    res_svr.actuals = [res_svr.actuals; y_scalar(teIdx)];
    res_svr.paths   = [res_svr.paths;   imgPaths(teIdx)];

    % ── Model 2 : GBT (Gradient Boosted Trees) ────────────────────────────
    fprintf('│  [GBT]  Training (500 cycles) …\n');
    gbtSin = fitrensemble(X_tr_z, ySin_tr, 'Method','LSBoost','NumLearningCycles',500,'LearnRate',0.05,'Learners',templateTree('MinLeafSize',5));
    gbtCos = fitrensemble(X_tr_z, yCos_tr, 'Method','LSBoost','NumLearningCycles',500,'LearnRate',0.05,'Learners',templateTree('MinLeafSize',5));
    
    pSinB = predict(gbtSin, X_te_z); pCosB = predict(gbtCos, X_te_z);
    angB  = atan2(pSinB, pCosB); angB(angB < 0) = angB(angB < 0) + 2*pi;
    res_boost.guesses = [res_boost.guesses; (angB/(2*pi))*24];
    res_boost.actuals = [res_boost.actuals; y_scalar(teIdx)];
    res_boost.paths   = [res_boost.paths;   imgPaths(teIdx)];

    % ── Model 3 : Random Forest ───────────────────────────────────────────
    fprintf('│  [RF]   Training (500 trees) …\n');
    rfSin = TreeBagger(500, X_tr_z, ySin_tr, 'Method','regression','MinLeafSize',5);
    rfCos = TreeBagger(500, X_tr_z, yCos_tr, 'Method','regression','MinLeafSize',5);
    
    pSinR = predict(rfSin, X_te_z); pCosR = predict(rfCos, X_te_z);
    angR  = atan2(pSinR, pCosR); angR(angR < 0) = angR(angR < 0) + 2*pi;
    res_rf.guesses = [res_rf.guesses; (angR/(2*pi))*24];
    res_rf.actuals = [res_rf.actuals; y_scalar(teIdx)];
    res_rf.paths   = [res_rf.paths;   imgPaths(teIdx)];

    % ── Ensemble ──────────────────────────────────────────────────────────
    ensSin = (pSinS + pSinB + pSinR) / 3;
    ensCos = (pCosS + pCosB + pCosR) / 3;
    angE   = atan2(ensSin, ensCos); angE(angE < 0) = angE(angE < 0) + 2*pi;
    res_ens.guesses = [res_ens.guesses; (angE/(2*pi))*24];
    res_ens.actuals = [res_ens.actuals; y_scalar(teIdx)];
    res_ens.paths   = [res_ens.paths;   imgPaths(teIdx)];

    writeFoldPredictions(fold, imgPaths(teIdx), y_scalar(teIdx), ...
                         (angS/(2*pi))*24, (angB/(2*pi))*24, (angR/(2*pi))*24, (angE/(2*pi))*24);
    fprintf('└───────────────────────────────────────────────\n\n');
end

% ── Step 4 : Summary ──────────────────────────────────────────────────────
fprintf('[4/5]  Summary\n');
fprintf('═══════════════════════════════════════════════════\n');
printModelResults('SVR (Support Vector)', res_svr);
printModelResults('GBT (Boosted Trees)',  res_boost);
printModelResults('RF  (Random Forest)',   res_rf);
printModelResults('Ensemble (SVR+GBT+RF)', res_ens);
fprintf('═══════════════════════════════════════════════════\n\n');

% ── Step 5 : Final Results ────────────────────────────────────────────────
allErr = arrayfun(@(g,a) circularTimeDiff(g,a)*60, res_ens.guesses, res_ens.actuals);
T = table((1:numel(allErr))', arrayfun(@hoursToHHMM, res_ens.guesses, 'UniformOutput', false), ...
          arrayfun(@hoursToHHMM, res_ens.actuals, 'UniformOutput', false), allErr, res_ens.paths, ...
          'VariableNames', {'Index','Guess','Actual','AbsErr_min','ImagePath'});
writetable(T, 'sky_time_results_m2.csv');
fprintf('Results saved to sky_time_results_m2.csv\n\n');

% ── Step 6 : Visualizations ───────────────────────────────────────────────
fprintf('[6/5]  Generating Visualizations …\n');
% Ensure the script's directory is on the path so we can call it
addpath(fileparts(mfilename('fullpath')));
visualize_m2_results;
fprintf('       Visualizations completed and saved in matlab_method_2/\n\n');

% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function feat = extractClassicalFeaturesChunked(imgPaths, inputSize, chunkSize)
    N = numel(imgPaths); feat = [];
    for s = 1:chunkSize:N
        e = min(s + chunkSize - 1, N);
        feat = [feat; extractClassicalFeatures(imgPaths(s:e), inputSize)]; %#ok<AGROW>
    end
end

function s = fileStem(p), [~, s, ~] = fileparts(p); end

function printModelResults(name, res)
    errs = arrayfun(@(g,a) circularTimeDiff(g,a)*60, res.guesses, res.actuals);
    fprintf('  %-26s  RMSE: %s   MAE: %s\n', name, minsToHHMM(sqrt(mean(errs.^2))), minsToHHMM(mean(errs)));
end

function str = minsToHHMM(mins)
    hh = floor(mins/60); mm = round(mod(mins,60));
    if mm==60, hh=hh+1; mm=0; end
    str = sprintf('%02dh%02dm', hh, mm);
end

function writeFoldPredictions(fold, paths, actualsHr, svrG, gbtG, rfG, ensG)
    n = numel(paths);
    % Calculate individual model errors in minutes
    errS = arrayfun(@(g,a) circularTimeDiff(g,a)*60, svrG, actualsHr);
    errB = arrayfun(@(g,a) circularTimeDiff(g,a)*60, gbtG, actualsHr);
    errR = arrayfun(@(g,a) circularTimeDiff(g,a)*60, rfG, actualsHr);
    errE = arrayfun(@(g,a) circularTimeDiff(g,a)*60, ensG, actualsHr);

    T = table(cellfun(@(p) [fileStem(p),'.jpg'], paths, 'UniformOutput', false), ...
              arrayfun(@hoursToHHMM, actualsHr, 'UniformOutput', false), ...
              arrayfun(@hoursToHHMM, svrG, 'UniformOutput', false), ...
              arrayfun(@hoursToHHMM, gbtG, 'UniformOutput', false), ...
              arrayfun(@hoursToHHMM, rfG,  'UniformOutput', false), ...
              arrayfun(@hoursToHHMM, ensG, 'UniformOutput', false), ...
              errS, errB, errR, errE, ...
              'VariableNames', {'PhotoName','Actual','SVR_Guess','GBT_Guess','RF_Guess','Ens_Guess',...
                                'SVR_Err','GBT_Err','RF_Err','Ens_Err'});
    writetable(T, sprintf('fold_%d_predictions.csv', fold));
end
