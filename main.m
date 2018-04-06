%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'C:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% serious parameters
k = 400;
training_image_count_per_category = 10;
verbose = false;

%% baby parameters
k = 40;
training_image_count_per_category = 4;
verbose = true;

run_experiment(training_image_count_per_category, k, verbose);