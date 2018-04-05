%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'C:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% baby parameters
k = 40;
training_image_count_per_category = 4;
verbose = true;

%% serious parameters
k = 400;
training_image_count_per_category = 10;
verbose = false;

%% 2.1 feature extraction
categories = {'airplanes', 'cars', 'faces', 'motorbikes'};

% collect descriptors from n images from each category
all_descriptors = extract_all_descriptors(categories, training_image_count_per_category);
fprintf("extracted %i features\n", size(all_descriptors,2));

%% 2.2 construct visual vocabulary (kmeans)
[C, idx] = vl_ikmeans(all_descriptors, k);
disp("constructed visual vocabulary");

if verbose
    % display some centroids
    figure(1);
    for i = 1:9
        subplot(3,3,i);
        descriptor = C(:,i);
        vl_plotsiftdescriptor(descriptor);
    end
end


%% collect all training and test data
testing_image_count_per_category = 50; % 50 = all test images
[training_labels, training_histograms] = get_labeled_histograms_from_set(categories, 'train', training_image_count_per_category, training_image_count_per_category, C);
[testing_labels, testing_histograms] = get_labeled_histograms_from_set(categories, 'test', testing_image_count_per_category, 0, C);
disp("collected training and test data");

%% train 4-class SVM just to test accuracy
model = train(training_labels, sparse(training_histograms), '-s 0');
predictions = predict(testing_labels, sparse(testing_histograms), model);

