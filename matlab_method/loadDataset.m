function [imgPaths, timeLabels, dateFeat] = loadDataset(datasetPath, formats)
% LOADDATASET  Scan a folder for images and extract DateTime from EXIF.
%
%   [imgPaths, timeLabels, dateFeat] = loadDataset(datasetPath, formats)
%
%   Inputs
%     datasetPath  – path to folder containing images
%     formats      – cell array of glob patterns, e.g. {'*.jpg','*.heic'}
%
%   Outputs
%     imgPaths    – Nx1 cell of absolute file paths
%     timeLabels  – Nx1 double, time-of-day in fractional hours  [0, 24)
%     dateFeat    – Nx4 double, circular date encoding per image
%                   cols: [sin(doy), cos(doy), sin(month), cos(month)]
%
%   Notes
%     • Only images whose EXIF contains a parseable 'DateTime' field are kept.
%     • HEIC requires MATLAB R2022a or newer (Image Processing Toolbox).
%     • DNG requires the Image Processing Toolbox.

    allPaths   = {};
    for k = 1:numel(formats)
        hits = dir(fullfile(datasetPath, formats{k}));
        for j = 1:numel(hits)
            allPaths{end+1} = fullfile(hits(j).folder, hits(j).name); %#ok<AGROW>
        end
    end

    if isempty(allPaths)
        error('No images found in "%s" matching the given formats.', datasetPath);
    end
    
    % Remove duplicates caused by case-insensitive file systems (e.g., .heic and .HEIC)
    allPaths = unique(allPaths, 'stable');

    imgPaths   = {};
    timeLabels = [];
    dateFeat   = [];

    fprintf('       Parsing %d candidate files …\n', numel(allPaths));

    for i = 1:numel(allPaths)
        fpath = allPaths{i};
        try
            dt = readDateTime(fpath);
            if isempty(dt)
                continue
            end

            % ── Time label (fractional hours) ─────────────────────────────
            h   = dt(4) + dt(5)/60 + dt(6)/3600;
            if h < 0 || h >= 24
                continue
            end

            % ── Date features (circular encoding) ─────────────────────────
            dv    = datevec(datenum(dt(1),dt(2),dt(3)));
            doy   = day(datetime(dt(1),dt(2),dt(3)), 'dayofyear');  % 1–366
            doyN  = (doy - 1) / 365;
            monN  = (dt(2) - 1) / 11;
            feat  = [sin(2*pi*doyN), cos(2*pi*doyN), ...
                     sin(2*pi*monN), cos(2*pi*monN)];

            imgPaths{end+1,1} = fpath;   %#ok<AGROW>
            timeLabels(end+1,1) = h;     %#ok<AGROW>
            dateFeat(end+1,:)   = feat;  %#ok<AGROW>

        catch ME
            warning('loadDataset: skipping %s — %s', fpath, ME.message);
        end
    end

    if isempty(imgPaths)
        error('No images with valid DateTime metadata found.');
    end
end

% ─────────────────────────────────────────────────────────────────────────
function dt = readDateTime(fpath)
% READDATETIME  Extract datetime vector [Y M D h m s] from EXIF.
%   Returns empty if no parseable DateTime tag is found.

    dt = [];
    
    % MATLAB's imfinfo often misses EXIF DateTime for .heic files.
    % We can leverage Python's PIL and pillow_heif as a reliable workaround.
    [~, ~, ext] = fileparts(fpath);
    if strcmpi(ext, '.heic')
        try
            py.pillow_heif.register_heif_opener();
            img = py.PIL.Image.open(fpath);
            exif_data = img.getexif();
            
            % 36867 is DateTimeOriginal, 306 is DateTime
            dt_py = exif_data.get(int32(36867));
            if isempty(dt_py) || isa(dt_py, 'py.NoneType')
                dt_py = exif_data.get(int32(306));
            end
            
            if ~isempty(dt_py) && ~isa(dt_py, 'py.NoneType')
                parsed = parseDateTimeString(char(dt_py));
                if ~isempty(parsed)
                    dt = parsed;
                    return;
                end
            end
        catch
            % Fallback to imfinfo if Python integration fails
        end
    end

    % imfinfo works for jpg/png/tiff/dng; HEIC needs R2022a+
    try
        info = imfinfo(fpath);

        % Gather candidate DateTime strings in priority order
        candidates = {};
        if isfield(info, 'DigitalCamera')
            dc = info.DigitalCamera;
            if isfield(dc, 'DateTimeOriginal') && ~isempty(dc.DateTimeOriginal)
                candidates{end+1} = dc.DateTimeOriginal;
            end
            if isfield(dc, 'DateTimeDigitized') && ~isempty(dc.DateTimeDigitized)
                candidates{end+1} = dc.DateTimeDigitized;
            end
        end
        if isfield(info, 'DateTime') && ~isempty(info.DateTime)
            candidates{end+1} = info.DateTime;
        end
    
        % Try to parse each candidate
        for k = 1:numel(candidates)
            parsed = parseDateTimeString(candidates{k});
            if ~isempty(parsed)
                dt = parsed;
                return
            end
        end
    catch
        % imfinfo might fail on certain HEIC files or corrupted headers
    end

    % Fallback: Use file modification date if EXIF is missing or imfinfo fails
    fileInfo = dir(fpath);
    if ~isempty(fileInfo)
        dt = datevec(fileInfo(1).datenum);
    end
    end

% ─────────────────────────────────────────────────────────────────────────
function dv = parseDateTimeString(s)
% PARSEDATETIMESTRING  Parse EXIF DateTime string 'YYYY:MM:DD HH:MM:SS'.
%   Returns [Y M D h m s] or empty on failure.

    dv = [];
    if isempty(s) || ~ischar(s)
        return
    end
    % Remove extra whitespace
    s = strtrim(s);
    % Standard EXIF format: '2025:09:05 08:03:07'
    tok = regexp(s, ...
        '^(\d{4}):(\d{2}):(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$','tokens');
    if isempty(tok)
        % Try ISO format as fallback: '2025-09-05T08:03:07'
        tok = regexp(s, ...
            '^(\d{4})-(\d{2})-(\d{2})[T\s](\d{2}):(\d{2}):(\d{2})','tokens');
    end
    if isempty(tok)
        return
    end
    nums = cellfun(@str2double, tok{1});
    if any(isnan(nums)) || nums(2) < 1 || nums(2) > 12 || ...
       nums(3) < 1 || nums(3) > 31 || ...
       nums(4) < 0 || nums(4) > 23
        return
    end
    dv = nums;
end
