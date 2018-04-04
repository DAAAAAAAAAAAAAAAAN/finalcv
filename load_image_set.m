function images = load_image_set(category, type, max, offset)
    images_dir = dir(fullfile('Caltech4', 'ImageData', strcat(category, '_', type), 'img*'));
    if length(images_dir) > max
        images_dir = images_dir(1+offset:max+offset, :);
    end
    
    n = length(images_dir);
    images = cell(1, n);
    for i = 1:n
        images{i} = imread(fullfile(images_dir(i).folder, images_dir(i).name));
    end
end