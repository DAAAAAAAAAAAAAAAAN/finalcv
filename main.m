%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')

%% 2.1 feature extraction
categories = {'airplanes', 'cars', 'faces', 'motorbikes'};
image_count = 3;


% collect descriptors from all images from all categories
descriptors = cell(1, length(categories)*image_count);
for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'train', image_count);
    
    % collect descriptors from all images densely
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 10); % default sift at the moment
        descriptors{(i-1)*image_count + j} = d;
    end
end
disp("extracted features");

%% 2.2 construct visual vocabulary
% run kmeans to get visual words
k = 400;
all_descriptors = cat(2, descriptors{:});
size(all_descriptors)
[idx, C] = kmeans(double(all_descriptors'), k);
disp("constructed visual vocabulary");

%% 2.3 + 2.4 quantization + histogram
for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'train', image_count);
    
    % represent each image as histogram
    for j = 1:length(images)
        image = images{i};
        [f, d] = vl_phow(im2single(image), 'Step', 10); % default sift at the moment
        % assign each descriptor to cluster
        idx = dsearchn(C, double(d'));
        h = histcounts(idx, k, 'BinMethod','integers', 'Normalization', 'probability');
    end
end
