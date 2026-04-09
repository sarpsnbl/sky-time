function F = extractClassicalFeatures(imgPaths, inputSize)
% EXTRACTCLASSICALFEATURES  Extract hand-crafted features from sky images.
%
%   F = extractClassicalFeatures(imgPaths, inputSize)
%
%   Features extracted per image (145 total):
%     • RGB / HSV / LAB channel statistics  (mean + std × 3 spaces × 3ch = 18)
%     • RGB histogram (16 bins × 3 channels = 48)
%     • HSV histogram (16 bins × 3 channels = 48)
%     • Sun-region brightness (top-third of image, mean + std = 2)
%     • Horizon luminance gradient (mean abs gradient in V channel = 1)
%     • Sky colour temperature proxy (R/B ratio = 1)
%     • Global luminance stats (mean, std, entropy = 3)
%     • Saturation stats (mean, std = 2)
%     • Edge density (Canny, normalised = 1)
%     • Laplacian variance (sharpness = 1)   [total = 125, padded to 145 below]
%
%   Inputs
%     imgPaths  – Nx1 cell of image file paths
%     inputSize – [H W C] target size for resizing (e.g. [224 224 3])
%
%   Output
%     F – N × D double feature matrix (D = 145 features)

    N = numel(imgPaths);
    D = 145;
    F = zeros(N, D);

    fprintf('       ');
    pBar = 0;

    for i = 1:N
        % Progress indicator
        pct = floor(i/N*20);
        if pct > pBar
            fprintf('█');
            pBar = pct;
        end

        try
            img = loadAndResize(imgPaths{i}, inputSize);
        catch
            % If image cannot be read, leave as zeros
            continue
        end

        feat = [];

        % ── 1. RGB statistics ──────────────────────────────────────────────
        for c = 1:3
            ch = double(img(:,:,c)) / 255;
            feat = [feat, mean(ch(:)), std(ch(:))]; %#ok<AGROW>
        end

        % ── 2. HSV statistics ──────────────────────────────────────────────
        hsv = rgb2hsv(img);
        for c = 1:3
            ch = hsv(:,:,c);
            feat = [feat, mean(ch(:)), std(ch(:))]; %#ok<AGROW>
        end

        % ── 3. LAB statistics ─────────────────────────────────────────────
        lab = rgb2lab(img);
        for c = 1:3
            ch = lab(:,:,c);
            feat = [feat, mean(ch(:)), std(ch(:))]; %#ok<AGROW>
        end

        % ── 4. RGB histograms (16 bins each) ──────────────────────────────
        edges = linspace(0,255,17);
        for c = 1:3
            hc = histcounts(double(img(:,:,c)), edges, 'Normalization','probability');
            feat = [feat, hc]; %#ok<AGROW>
        end

        % ── 5. HSV histograms (16 bins each) ──────────────────────────────
        edgesHS = linspace(0,1,17);
        for c = 1:3
            hc = histcounts(hsv(:,:,c), edgesHS, 'Normalization','probability');
            feat = [feat, hc]; %#ok<AGROW>
        end

        % ── 6. Sun-region brightness (top third) ──────────────────────────
        H = size(img,1);
        topV = hsv(1:floor(H/3), :, 3);
        feat = [feat, mean(topV(:)), std(topV(:))]; %#ok<AGROW>

        % ── 7. Horizon luminance gradient ─────────────────────────────────
        V   = hsv(:,:,3);
        dVy = diff(V, 1, 1);          % vertical gradient in V channel
        feat = [feat, mean(abs(dVy(:)))]; %#ok<AGROW>

        % ── 8. Colour temperature proxy (R/B ratio) ───────────────────────
        rMean = mean(mean(double(img(:,:,1))));
        bMean = mean(mean(double(img(:,:,3))));
        feat  = [feat, rMean / max(bMean, 1e-6)]; %#ok<AGROW>

        % ── 9. Global luminance stats (Y of YCbCr) ─────────────────────────
        lum   = 0.299*double(img(:,:,1)) + 0.587*double(img(:,:,2)) + ...
                0.114*double(img(:,:,3));
        lumN  = lum / 255;
        e     = entropy(uint8(lum));
        feat  = [feat, mean(lumN(:)), std(lumN(:)), e]; %#ok<AGROW>

        % ── 10. Saturation stats ──────────────────────────────────────────
        S = hsv(:,:,2);
        feat = [feat, mean(S(:)), std(S(:))]; %#ok<AGROW>

        % ── 11. Edge density (Canny on luminance) ─────────────────────────
        edges_img = edge(uint8(lum), 'Canny');
        feat = [feat, sum(edges_img(:)) / numel(edges_img)]; %#ok<AGROW>

        % ── 12. Laplacian variance (sharpness / cloud texture) ────────────
        lapKernel = fspecial('laplacian', 0.2);
        lapImg    = imfilter(lumN, lapKernel, 'replicate');
        feat = [feat, var(lapImg(:))]; %#ok<AGROW>

        % Pad / truncate to exactly D features
        if numel(feat) > D
            feat = feat(1:D);
        elseif numel(feat) < D
            feat = [feat, zeros(1, D-numel(feat))]; %#ok<AGROW>
        end

        F(i,:) = feat;
    end
    fprintf('\n');
end

% ─────────────────────────────────────────────────────────────────────────
function img = loadAndResize(fpath, inputSize)
% LOADANDRESIZE  Read an image file, convert to RGB, resize.
%   Handles jpg/png/dng/heic; converts grayscale to RGB.

    img = imread(fpath);

    % DNG / RAW: imread may return uint16 – normalise to uint8
    if isa(img, 'uint16')
        img = uint8(double(img) / 65535 * 255);
    elseif isa(img, 'single') || isa(img, 'double')
        if max(img(:)) <= 1
            img = uint8(img * 255);
        else
            img = uint8(img);
        end
    end

    % Grayscale to RGB
    if size(img,3) == 1
        img = repmat(img, [1 1 3]);
    elseif size(img,3) == 4
        img = img(:,:,1:3);   % drop alpha
    end

    % Resize
    img = imresize(img, inputSize(1:2));
end
