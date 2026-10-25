function minutesPred = predictCNN(net, imgPaths, dateFeat, cfg)
% PREDICTCNN  Run inference and decode cyclic output to minutes-of-day.
%
%   The network outputs [sin(t), cos(t)] (2-D regression target).
%   This function converts those back to clock minutes in [0, 1440).
%
%   Inputs
%     net      – trained SeriesNetwork / DAGNetwork
%     imgPaths – cell array of image paths
%     dateFeat – M×D numeric date feature matrix
%     cfg      – configuration struct (needs cfg.inputSize, cfg.miniBatch)
%
%   Output
%     minutesPred – M×1 vector of predicted minutes in [0, 1440)

    MINUTES_PER_DAY = 1440.0;
    N = numel(imgPaths);

    % ── Build inference datastore (no augmentation, no labels) ─────────
    % Same index-based pattern as buildDatastore to avoid the
    % "more than one observation per row" error from arrayDatastore.
    idxDS = arrayDatastore((1:N)', 'IterationDimension', 1);
    predDS = transform(idxDS, ...
        @(idx) readInferenceSample(idx, imgPaths, dateFeat, cfg.inputSize(1:2)));

    % ── Predict ─────────────────────────────────────────────────────────
    rawPred = predict(net, predDS, 'MiniBatchSize', cfg.miniBatch);
    % rawPred is M×2 : [sin(t), cos(t)]

    % ── Decode cyclic output → minutes ──────────────────────────────────
    angles = atan2(rawPred(:,1), rawPred(:,2));
    angles(angles < 0) = angles(angles < 0) + 2 * pi;
    minutesPred = angles * MINUTES_PER_DAY / (2 * pi);
end

function out = readInferenceSample(idxCell, imgPaths, dateFeat, targetHW)
% Returns {img, feat} — no label needed for inference.
% feat is D×1 (column vector) to match featureInputLayer expectations.
    if iscell(idxCell)
        i = idxCell{1};
    else
        i = idxCell;
    end
    i    = i(1);
    img  = loadAndNormalize(imgPaths{i}, targetHW);
    feat = single(dateFeat(i, :)');   % transpose → D×1
    out  = {img, feat};
end

function img = loadAndNormalize(path, targetHW)
    try
        img = imread(path);
    catch
        img = zeros(targetHW(1), targetHW(2), 3, 'uint8');
    end
    if size(img,3) == 1,  img = repmat(img,1,1,3); end
    if size(img,3) == 4,  img = img(:,:,1:3);      end
    img = im2single(imresize(img, targetHW));
    mean_rgb = reshape([0.485,0.456,0.406],1,1,3);
    std_rgb  = reshape([0.229,0.224,0.225],1,1,3);
    img = (img - mean_rgb) ./ std_rgb;
end
