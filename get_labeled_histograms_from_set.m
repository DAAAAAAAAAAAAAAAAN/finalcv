function [labels, histograms] = get_labeled_histograms_from_set (categories, image_set, n, offset, C)
    k = size(C, 2);
    labels = zeros([length(categories) * n, 1]);
    histograms = zeros([length(categories) * n, k]);

    for i = 1:length(categories)
        category = categories{i};

        % load list of images from this set
        images = load_image_set(category, image_set, n, offset);

        % represent each image as histogram
        for j = 1:length(images)
            image = images{j};

            h = extract_histogram(image, C);

            index = (i-1)*n + j;
            labels(index) = i;
            histograms(index, :) = h;
        end
    end
end