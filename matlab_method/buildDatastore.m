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

    % Capture the main thread's Python path (background workers often lose this)
    try
        pyPathList = cellfun(@char, cell(py.sys.path), 'UniformOutput', false);
    catch
        pyPathList = {};
    end

    % ── Image datastore ───────────────────────────────────────────────
    imds = imageDatastore(imgPaths, ...
        'ReadFcn', @(p) readAndPreprocess(p, cfg.inputSize, augment, pyPathList));

    % ── Feature datastore (date) ──────────────────────────────────────
    % ── Feature datastore (date) ──────────────────────────────────────
    % Transpose the Nx4 matrix to 4xN and iterate along columns (Dim 2)
    % so it outputs 4x1 column vectors instead of 1x4 row vectors.
    featDS = arrayDatastore(single(dateFeat)', 'IterationDimension', 2, ...
                            'OutputType','cell');

    % ── Label datastore ───────────────────────────────────────────────
    labelDS = arrayDatastore(single(timeLabels), 'IterationDimension', 1, ...
                             'OutputType','cell');

    % ── Combine: order must match alphabetical input-layer names ──────
    ds = combine(imds, featDS, labelDS);
end

% ─────────────────────────────────────────────────────────────────────────
function img = readAndPreprocess(fpath, inputSize, augment, pyPathList)
% READANDPREPROCESS  Load an image, convert to float [0,1], optionally augment.

    if nargin < 4
        pyPathList = {};
    end

    fpath = char(fpath);

    % Bypass imread entirely for HEIC to prevent missing-addon crashes
    [~, ~, ext] = fileparts(fpath);
    if strcmpi(ext, '.heic') || strcmpi(ext, '.heif')
        try
            % Inject the main thread's Python path into this worker's environment
            if ~isempty(pyPathList)
                sys = py.importlib.import_module('sys');
                currPaths = cellfun(@char, cell(sys.path), 'UniformOutput', false);
                for k = 1:numel(pyPathList)
                    if ~any(strcmp(currPaths, pyPathList{k}))
                        sys.path.append(pyPathList{k});
                    end
                end
            end

            py.pillow_heif.register_heif_opener();
            img_py = py.PIL.Image.open(fpath).convert('RGB');
            w = double(img_py.width);
            h = double(img_py.height);
            % Convert Python bytes -> py.array -> MATLAB uint8
            b = py.array.array('B', img_py.tobytes());
            raw = permute(reshape(uint8(b), [3, w, h]), [3, 2, 1]);
        catch pyErr
            error('Python failed to load HEIC "%s". Details: %s', fpath, pyErr.message);
        end
    else
        raw = imread(fpath);
    end

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
        
        % Random rotation [-10, 10] degrees
        if rand > 0.5
            angle = (rand * 20) - 10;
            img = imrotate(img, angle, 'bilinear', 'crop');
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