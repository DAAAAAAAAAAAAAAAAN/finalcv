%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'C:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% baby parameters
k = 40;
bow_image_count = 4;

%% serious parameters
k = 400;
bow_image_count = 10;

%% 2.1 feature extraction
categories = {'airplanes', 'cars', 'faces', 'motorbikes'};

% collect descriptors from all images from all categories
descriptors = cell(1, length(categories)*bow_image_count);
for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'train', bow_image_count, 0);
    
    % collect descriptors from all images densely
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
        descriptors{(i-1)*bow_image_count + j} = d;
    end
end
all_descriptors = cat(2, descriptors{:});
fprintf("extracted %i features\n", size(all_descriptors,2));

%% 2.2 construct visual vocabulary
% run kmeans to get visual words
[C, idx] = vl_ikmeans(all_descriptors, k);
%[idx, C] = kmeans(double(all_descriptors'), k);
disp("constructed visual vocabulary");

%% display some centroids
figure(1);
for i = 1:9
    subplot(3,3,i);
    descriptor = C(:,i);
    %descriptor = C(i,:)';
    vl_plotsiftdescriptor(descriptor);
end

%% 2.3 + 2.4 quantization + histogram
class_labels = zeros([length(categories) * bow_image_count, 1]);
histograms = zeros([length(categories) * bow_image_count, k]);

for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'train', bow_image_count, bow_image_count);
    
    % represent each image as histogram
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
        % assign each descriptor to cluster
        %idx = dsearchn(C, double(d'));
        idx = vl_ikmeanspush(d, C);
        %h = histcounts(idx, k, 'BinMethod', 'integers', 'Normalization', 'probability');
        h = vl_ikmeanshist(k, idx);
        
        index = (i-1)*bow_image_count + j;
        class_labels(index) = i;
        histograms(index, :) = h;
        
        % visualization
%         figure(2);
%         subplot(1,2,1);
%         histogram(idx, k);
%         subplot(1,2,2);
%         imshow(image);
%         waitforbuttonpress
    end
end


%% 2.5 directly train 4-class SVM classifier
model = train(class_labels, sparse(histograms), '-s 0');
disp("trained 4-class SVM");

%% analysis: assign images to class directly
pred_image_count = 50;
real_labels = zeros([length(categories) * pred_image_count, 1]);
histograms2 = zeros([length(categories) * pred_image_count, k]);

for i = 1:length(categories)
    category = categories{i};
    
    % load list of images from this set
    images = load_image_set(category, 'test', pred_image_count, 0);
    
    % represent each image as histogram
    for j = 1:length(images)
        image = images{j};
        [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
        % assign each descriptor to cluster
        %idx = dsearchn(C, double(d'));
        idx = vl_ikmeanspush(d, C);
        %h = histcounts(idx, k, 'BinMethod', 'integers', 'Normalization', 'probability');
        h = vl_ikmeanshist(k, idx);
        
        index = (i-1)*pred_image_count + j;
        real_labels(index) = i;
        histograms2(index, :) = h;
    end
end

predictions = predict(real_labels, sparse(histograms2), model);
disp("made predictions");
