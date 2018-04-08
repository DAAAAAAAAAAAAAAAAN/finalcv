%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'D:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% parameter sets
sift_methods = {'dense', 'keypoint'};
sift_types = {'gray', 'RGB', 'rgb', 'opponent'};
step_sizes = [10, 15, 20];
vocabulary_sizes = [400, 800, 1600, 2000, 4000];
numbers_of_training_samples = [4, 16, 64, 200];
number_of_testing_samples = 50; % all test images

%% experiments for training set size (32 is a decent balance and leaves room for improvement so used as default for next experiments)
for j = 1:length(numbers_of_training_samples)

    %% experiments for training set size (400 performed best so used as default for next experiments)
    for i = 1:length(vocabulary_sizes)
        run_experiment(sift_methods{1}, sift_types{1}, step_sizes(2), vocabulary_sizes(i), numbers_of_training_samples(j), number_of_testing_samples, false);
    end

    %% experiments for sift types
    for i = 1:length(sift_types)
        run_experiment(sift_methods{1}, sift_types{i}, step_sizes(2), vocabulary_sizes(1), numbers_of_training_samples(j), number_of_testing_samples, false);
    end

    %% experiments for sift method
    for i = 1:length(sift_methods)
        run_experiment(sift_methods{i}, sift_types{1}, step_sizes(2), vocabulary_sizes(1), numbers_of_training_samples(j), number_of_testing_samples, false);
    end

    %% experiments for step size
    for i = 1:length(step_sizes)
        run_experiment(sift_methods{1}, sift_types{1}, step_sizes(i), vocabulary_sizes(1), numbers_of_training_samples(j), number_of_testing_samples, false);
    end
end