function [labels, histograms, paths] = get_labeled_histograms_from_set (categories, image_set, n, offset, C)
    k = size(C, 2);
    labels = zeros([length(categories) * n, 1]);
    histograms = zeros([length(categories) * n, k]);
    paths = cell(1, length(categories) * n);

    for i = 1:length(categories)
        category = categories{i};

        % load list of images from this set
        [images, category_paths] = load_image_set(category, image_set, n, offset);

        % represent each image as histogram
        for j = 1:length(images)
            image = images{j};

            h = extract_histogram(image, C);

            index = (i-1)*n + j;
            labels(index) = i;
            histograms(index, :) = h;
            paths{index} = category_paths{j};
        end
    end
end