function preds = predictCNN(net, imgPaths, dateFeat, cfg)
% PREDICTCNN  Run inference with the trained multi-input network.
%
%   preds = predictCNN(net, imgPaths, dateFeat, cfg)
%
%   Returns an Nx1 vector of predicted times in fractional hours [0, 24).

    N     = numel(imgPaths);
    preds = zeros(N, 1);

    for i = 1:N
        % Load and preprocess image (no augmentation)
        try
            img = readImageForInference(imgPaths{i}, cfg.inputSize);
        catch ME
            warning('predictCNN: cannot read %s — %s. Using zero image.', ...
                    imgPaths{i}, ME.message);
            img = zeros(cfg.inputSize, 'single');
        end

        df   = single(dateFeat(i,:));    % 1×4

        % predict() with multi-input network:
        %   pass inputs in the same alphabetical order as the layer names
        %   'data' (image) then 'data_date' (features)
        raw  = predict(net, img, df);    % returns scalar

        % Clamp to [0, 24)
        preds(i) = max(0, min(23.9999, double(raw)));
    end
end

% ─────────────────────────────────────────────────────────────────────────
function img = readImageForInference(fpath, inputSize)
    raw = imread(fpath);
    if isa(raw,'uint16')
        raw = uint8(double(raw)/65535*255);
    elseif isa(raw,'single') || isa(raw,'double')
        raw = uint8(min(255, raw * (max(raw(:)) <= 1) * 255 + ...
                             raw * (max(raw(:))  > 1)));
    end
    if size(raw,3) == 1,    raw = repmat(raw,[1 1 3]); end
    if size(raw,3) == 4,    raw = raw(:,:,1:3);         end
    img = single(imresize(raw, inputSize(1:2))) / 255;
end
