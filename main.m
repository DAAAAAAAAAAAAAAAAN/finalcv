%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'C:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% serious parameters
k = 400;
training_image_count_per_category = 10;
verbose = false;

%% baby parameters
k = 400;
training_image_count_per_category = 3;
verbose = true;

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

%% construct training and test labels for 4 one-vs-all classifiers
classifier_labels = cell(length(categories), 2);
for i = 1:length(categories)
    classifier_labels{i, 1} = training_labels == i;
    classifier_labels{i, 2} = testing_labels == i;
end

%% train 4-class SVM just to test accuracy
fprintf("4-class classifier:  ");
model = train(training_labels, sparse(training_histograms), '-s 0 -q');
predictions = predict(testing_labels, sparse(testing_histograms), model);

%% for each category, train and run one-vs-all classifier
for i = 1:length(categories)
    category = categories{i};
    
    % train and run SVM model
    model = train(double(classifier_labels{i,1}), sparse(training_histograms), '-s 0 -q');
    [predictions, accuracy, decision_values] = predict(double(classifier_labels{i,2}), sparse(testing_histograms), model, '-q');
    
    % sort predictions by confidence. The sign of the decision values is
    % determined by the first prediction: https://stackoverflow.com/questions/11030253/decision-values-in-libsvm
    if predictions(1) == 1
        [sorted, idx] = sort(decision_values, 'descend');
    else
        [sorted, idx] = sort(decision_values, 'ascend');
    end
    sorted_labels = classifier_labels{i,2}(idx); % apply sorting to testing labels
    
    % calculate MAP score
    n = length(sorted_labels);
    mc = testing_image_count_per_category;
    x = double(squeeze(sorted_labels'));
    x(x==1) = (1:mc); % convert [1 0 1 1 0 1] to [1 0 2 3 0 4]
    MAP = (1/mc) * sum(x ./ (1:n));
    
    % print results
    fprintf("one-vs-all classifier for %s:  %.1f%% accuracy, %.2f MAP\n", category, accuracy(1), MAP);
end

