function all_descriptors = extract_all_descriptors (categories, n, sift_method, sift_type, step_size)
    descriptors = cell(1, length(categories)*n);
    for i = 1:length(categories)
        category = categories{i};

        % load list of images from this set
        images = load_image_set(category, 'train', n, 0);

        % collect descriptors from all images densely
        for j = 1:length(images)
            image = images{j};
            d = extract_descriptor(image, sift_method, sift_type, step_size);
            descriptors{(i-1)*n + j} = d;
        end
    end
    all_descriptors = cat(2, descriptors{:});
end