function ds = buildDatastore(imgPaths, dateFeat, labels, cfg, isTraining)
% BUILDDATASTORE  Build a combined datastore for multi-input CNN training.
%
%   For the training set (isTraining=true) a data-augmentation pipeline is
%   applied that matches the Python transforms:
%     • Random horizontal flip          (p = 0.5)
%     • Random rotation                 (±5°)
%     • Colour jitter                   (brightness 0.25, contrast 0.2,
%                                        saturation 0.25, hue 0.04)
%     • Random resized crop             (scale 0.85–1.0)
%     • ImageNet mean/std normalisation
%
%   For the validation set only resize + normalise are applied.
%
%   Inputs
%     imgPaths   – cell array of image file paths
%     dateFeat   – M×D numeric array of date/metadata features
%     labels     – M×2 numeric array of [sin(t), cos(t)] targets
%     cfg        – configuration struct (needs cfg.inputSize)
%     isTraining – logical flag
%
%   Output
%     ds – combined datastore ready for trainNetwork

    targetHW = cfg.inputSize(1:2);   % [H W]
    N        = numel(imgPaths);

    % ── Index-based datastore ───────────────────────────────────────────
    % Using a single arrayDatastore over row indices is the most reliable
    % way to feed multi-input networks in MATLAB.  The transform reads the
    % image AND assembles the feature/label cells in one step, so
    % trainNetwork always receives exactly one {img, feat, label} triple
    % per sample — no ambiguity about "observations per row".
    idxDS = arrayDatastore((1:N)', 'IterationDimension', 1);

    ds = transform(idxDS, ...
        @(idx) readOneSample(idx, imgPaths, dateFeat, labels, targetHW, isTraining));
end

function out = readOneSample(idxCell, imgPaths, dateFeat, labels, targetHW, isTraining)
% Called by the transform for every sample.  Returns a 1×3 cell:
%   {H×W×3 single image,  D×1 single feat col,  1×2 single label row}
%
%   IMPORTANT: featureInputLayer expects features as a column vector (D×1).
%   Returning a row (1×D) causes trainNetwork to see N columns as N separate
%   observations when it stacks the mini-batch — hence the error.

    % arrayDatastore may wrap the value in a cell or pass it directly
    if iscell(idxCell)
        i = idxCell{1};
    else
        i = idxCell;
    end
    i = i(1);   % guarantee scalar

    % ── Load image ──────────────────────────────────────────────────────
    img = loadAndResize(imgPaths{i}, targetHW);

    % ── Augment / normalise ─────────────────────────────────────────────
    if isTraining
        img = augmentImage(img, targetHW);
    else
        img = imnormalize(img);
    end

    % ── Features: D×1 column vector ─────────────────────────────────────
    feat = single(dateFeat(i, :)');   % transpose → D×1

    % ── Label: 1×2 row ──────────────────────────────────────────────────
    lbl  = single(labels(i, :));      % 1×2  [sin(t), cos(t)]

    out = {img, feat, lbl};
end


% ── Private helpers ────────────────────────────────────────────────────────

function img = loadAndResize(path, targetHW)
% Read image and enforce channel count (Resizing removed for speed)
    try
        img = imread(path);
    catch
        img = zeros(targetHW(1), targetHW(2), 3, 'uint8');
    end
    
    if size(img, 3) == 1
        img = repmat(img, 1, 1, 3);      % greyscale → RGB
    elseif size(img, 3) == 4
        img = img(:,:,1:3);              % drop alpha
    end
    
    % imresize is intentionally omitted here: main.m re-routes imgPaths to
    % the pre-resized dataset_224x224 folder, so all images are already the
    % correct spatial size.  randomResizedCrop (training only) still calls
    % imresize internally after cropping.
    img = imresize(img, targetHW);   % 224→112 (or whatever cfg.inputSize is)
    
    img = im2single(img);                % convert to [0,1] float32
end

function img = augmentImage(img, targetHW)
% Full augmentation pipeline for training images.

    % 1. Random horizontal flip
    if rand() > 0.5
        img = fliplr(img);
    end

    % 2. Random rotation ±5°
    angle = (rand() * 10) - 5;
    img   = imrotate(img, angle, 'bilinear', 'crop');

    % 3. Colour jitter (brightness, contrast, saturation, hue)
    % img = colorJitter(img, 0.25, 0.2, 0.25, 0.04);

    % 4. Random resized crop (scale 0.85–1.0)
    img = randomResizedCrop(img, targetHW, 0.85, 1.0);

    % 5. ImageNet normalisation
    img = imnormalize(img);
end


function img = imnormalize(img)
% Subtract ImageNet mean and divide by std (channel-wise).
    mean_rgb = reshape([0.485, 0.456, 0.406], 1, 1, 3);
    std_rgb  = reshape([0.229, 0.224, 0.225], 1, 1, 3);
    img = (img - mean_rgb) ./ std_rgb;
end

function img = colorJitter(img, brightness, contrast, saturation, hue)
% Apply random colour jitter similar to torchvision ColorJitter.
%   All factors are applied in a random order with random magnitudes.

    ops = randperm(4);
    for k = 1:4
        switch ops(k)
            case 1  % Brightness
                f   = 1 + (rand() * 2 - 1) * brightness;
                img = img * f;
            case 2  % Contrast
                f   = 1 + (rand() * 2 - 1) * contrast;
                mu  = mean(img(:));
                img = (img - mu) * f + mu;
            case 3  % Saturation
                hsv = rgb2hsv(img);
                f   = 1 + (rand() * 2 - 1) * saturation;
                hsv(:,:,2) = hsv(:,:,2) * f;
                img = hsv2rgb(hsv);
            case 4  % Hue shift
                hsv = rgb2hsv(img);
                delta = (rand() * 2 - 1) * hue;
                hsv(:,:,1) = mod(hsv(:,:,1) + delta, 1);
                img = hsv2rgb(hsv);
        end
    end
    img = single(max(0, min(1, img)));   % clamp to [0, 1]
end

function img = randomResizedCrop(img, targetHW, scaleMin, scaleMax)
% Crop a random sub-rectangle (area fraction in [scaleMin, scaleMax])
% then resize to targetHW. Mirrors torchvision RandomResizedCrop.

    [H, W, ~] = size(img);
    totalArea  = H * W;
    scale      = scaleMin + rand() * (scaleMax - scaleMin);
    cropArea   = totalArea * scale;

    % Keep aspect ratio within [3/4, 4/3]
    ratio      = 0.75 + rand() * (4/3 - 0.75);
    cropW      = min(W, round(sqrt(cropArea * ratio)));
    cropH      = min(H, round(sqrt(cropArea / ratio)));
    cropW      = max(1, cropW);
    cropH      = max(1, cropH);

    x0 = randi(max(1, W - cropW + 1));
    y0 = randi(max(1, H - cropH + 1));

    img = img(y0:y0+cropH-1, x0:x0+cropW-1, :);
    img = imresize(img, targetHW);
end
