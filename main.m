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
    images = load_image_set(category, 'train', image_count, 0);
    
    % collect descriptors from all images densely
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
        descriptors{(i-1)*image_count + j} = d;
    end
end
all_descriptors = cat(2, descriptors{:});
fprintf("extracted %i features\n", size(all_descriptors,2));

%% 2.2 construct visual vocabulary
% run kmeans to get visual words
k = 40;
% [C, idx] = vl_ikmeans(all_descriptors, k);
[idx, C] = kmeans(double(all_descriptors'), k);
disp("constructed visual vocabulary");

%% display some centroids
figure(1);
for i = 1:9
    subplot(3,3,i);
    %descriptor = C(:,i);
    descriptor = C(i,:)';
    vl_plotsiftdescriptor(descriptor);
end

%% 2.3 + 2.4 quantization + histogram
for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'train', image_count, 50);
    
    % represent each image as histogram
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
        % assign each descriptor to cluster
        idx = dsearchn(C, double(d'));
        h = histcounts(idx, k, 'BinMethod', 'integers', 'Normalization', 'probability');
        figure(2);
        subplot(1,2,1);
        histogram(idx, k);
        subplot(1,2,2);
        imshow(image);
        
        waitforbuttonpress
    end
end


