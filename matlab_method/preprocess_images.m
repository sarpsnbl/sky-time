% preprocess_images.m
clear; clc;

srcDir  = 'dataset';          % Your current folder with varying sizes
destDir = 'dataset_224x224';  % The new folder we are creating

% Make the destination folder if it doesn't exist
if ~exist(destDir, 'dir')
    mkdir(destDir);
end

% Get all images
formats = {'*.jpg','*.jpeg','*.png','*.dng','*.heic','*.HEIC'};
allFiles = {};
for k = 1:numel(formats)
    hits = dir(fullfile(srcDir, formats{k}));
    for j = 1:numel(hits)
        allFiles{end+1} = fullfile(hits(j).folder, hits(j).name); %#ok<AGROW>
    end
end

N = numel(allFiles);
fprintf('Found %d images. Starting batch resize to 224x224...\n', N);

% Process them all
for i = 1:N
    srcPath = allFiles{i};
    [~, name, ~] = fileparts(srcPath);
    
    destPath = fullfile(destDir, sprintf('%s.jpg', name));
    
    try
        img = imread(srcPath);
        
        % --- NEW: Fix EXIF Orientation before stripping metadata ---
        try
            info = imfinfo(srcPath);
            if isfield(info, 'Orientation') && ~isempty(info(1).Orientation)
                switch info(1).Orientation
                    case 2, img = fliplr(img);
                    case 3, img = rot90(img, 2);
                    case 4, img = flipud(img);
                    case 5, img = permute(img, [2 1 3]);
                    case 6, img = rot90(img, -1); % Physically rotate 90 CW
                    case 7, img = rot90(permute(img, [2 1 3]), -1);
                    case 8, img = rot90(img, 1);  % Physically rotate 90 CCW
                end
            end
        catch
            % If imfinfo fails to read metadata (like on some HEIC files), 
            % just continue with the raw image.
        end
        % -----------------------------------------------------------
        
        % Force 3 channels (RGB)
        if size(img, 3) == 1
            img = repmat(img, 1, 1, 3);
        elseif size(img, 3) == 4
            img = img(:,:,1:3);
        end
        
        % Force exactly 224x224
        img_resized = imresize(img, [224 224]);
        
        % Write to new folder
        imwrite(img_resized, destPath, 'Quality', 100);
        
    catch ME
        fprintf('Failed to process %s: %s\n', name, ME.message);
    end
    
    % Print progress every 100 images
    if mod(i, 100) == 0
        fprintf('  Processed %d / %d...\n', i, N);
    end
end

fprintf('\nDone! Point main.m to "%s"\n', destDir);