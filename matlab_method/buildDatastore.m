function ds = buildDatastore(imgPaths, dateFeat, timeLabels, cfg, augment)
% BUILDDATASTORE  Create a combined datastore for multi-input trainNetwork.
%
%   The combined datastore returns one observation at a time as:
%       {image [H×W×3 uint8], dateFeat [1×4 single], timeLabel [1×1 single]}
%
%   Input layer order in the network (alphabetical by layer Name):
%       'data'      ← image  (SqueezeNet default name)
%       'data_date' ← date features
%   So combine() order must be:  imds, featureDS, labelDS
%
%   Inputs
%     imgPaths   – Nx1 cell of file paths
%     dateFeat   – N×4 date feature matrix
%     timeLabels – Nx1 time in fractional hours
%     cfg        – configuration struct
%     augment    – logical; if true, apply data augmentation

    % ── Image datastore ───────────────────────────────────────────────
    imds = imageDatastore(imgPaths, ...
        'ReadFcn', @(p) readAndPreprocess(p, cfg.inputSize, augment));

    % ── Feature datastore (date) ──────────────────────────────────────
    featDS = arrayDatastore(single(dateFeat), 'IterationDimension', 1, ...
                            'OutputType','same');

    % ── Label datastore ───────────────────────────────────────────────
    labelDS = arrayDatastore(single(timeLabels), 'IterationDimension', 1, ...
                             'OutputType','same');

    % ── Combine: order must match alphabetical input-layer names ──────
    ds = combine(imds, featDS, labelDS);
end

% ─────────────────────────────────────────────────────────────────────────
function img = readAndPreprocess(fpath, inputSize, augment)
% READANDPREPROCESS  Load an image, convert to float [0,1], optionally augment.

    % Read
    raw = imread(fpath);

    % Normalise non-uint8 formats (DNG / RAW)
    if isa(raw,'uint16')
        raw = uint8(double(raw)/65535*255);
    elseif isa(raw,'single') || isa(raw,'double')
        raw = uint8(raw * (max(raw(:)) <= 1) * 255 + ...
                    raw * (max(raw(:))  > 1));
    end

    % Force RGB
    if size(raw,3) == 1
        raw = repmat(raw,[1 1 3]);
    elseif size(raw,3) == 4
        raw = raw(:,:,1:3);
    end

    % Resize
    img = imresize(raw, inputSize(1:2));

    % ── Augmentation (training only) ──────────────────────────────────
    if augment
        % Random horizontal flip
        if rand > 0.5
            img = fliplr(img);
        end
        % Random brightness jitter ±10 %
        jitter = 1 + (rand*0.2 - 0.1);
        img    = uint8(min(255, double(img)*jitter));
        % Random colour temperature shift (slight R↔B balance change)
        rShift = 1 + (rand*0.1 - 0.05);
        bShift = 1 + (rand*0.1 - 0.05);
        imgD   = double(img);
        imgD(:,:,1) = min(255, imgD(:,:,1)*rShift);
        imgD(:,:,3) = min(255, imgD(:,:,3)*bShift);
        img = uint8(imgD);
    end

    % Convert to single [0,1] – required by regressionLayer
    img = single(img) / 255;
end
