heicFiles = dir(fullfile('dataset', '**', '*.heic'));
for i = 1:numel(heicFiles)
    p = fullfile(heicFiles(i).folder, heicFiles(i).name);
    img = imread(p);
    newPath = strrep(p, '.heic', '.jpg');
    imwrite(img, newPath, 'Quality', 95);
    delete(p);
    sprintf('Converted: %s', newPath, 'images left: %d', numel(heicFiles) - i)
end