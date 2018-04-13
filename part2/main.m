%% Daan Smedinga & Jens Dudink 
clc
run 'matconvnet/matlab/vl_setupnn' ;

% Note: we assume liblinear and matconvnet (renamed as above) to be 
% in the same folder as part 2 of the assignment.


%% fine-tune cnn
% The fine-tuned versions are all saved in data file. 

addpath('matconvnet/matlab')
addpath('liblinear-2.1/matlab/')

[net, info, expdir] = finetune_cnn();
disp('complete')
%% Save file

save(fullfile(expdir, 'b100_e80.mat'), 'net');
disp('succesfull')
%% extract features and train svm
batch_sizes = [ 50, 100 ]; 
epoch_sizes = [ 40, 80, 120 ]; 

addpath('matconvnet/matlab')
addpath('liblinear-2.1/matlab/')
dir = 'data';
results = {};
counter = 1;
for b_size = batch_sizes
    for e_size = epoch_sizes
        % Load finetuned net
        nets.fine_tuned = load(fullfile(dir, strcat('b', num2str(b_size),'_e', num2str(e_size),'.mat'))); 
        nets.fine_tuned = nets.fine_tuned.net;
        
        % Load pretrained net
        nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); 
        nets.pre_trained = nets.pre_trained.net; 
        
        % Load data
        data = load(fullfile(expdir, 'imdb-caltech.mat'));
        disp('test')
        
        % Save svm results
        results{counter} = train_svm(nets, data);
        counter = counter + 1;
    end
end

disp('Training complete')

%% Show results
batch_sizes = [ 50, 100 ]; 
epoch_sizes = [ 40, 80, 120 ]; 

addpath('matconvnet/matlab')
addpath('liblinear-2.1/matlab/')
dir = 'data';
counter = 1;
for b_size = batch_sizes
    for e_size = epoch_sizes
        disp('CNN: fine_tuned_accuracy   SVM: pre_trained_accuracy:  Fine_tuned_accuracy:')
        disp(strcat('b',num2str(b_size),'_e', num2str(e_size)))
        
        % Show results
        results{counter}
        
        counter = counter + 1;
    end
end

disp('End of results')

%% t-SNE
addpath('tSNE_matlab');
addpath('data');
clc
dir = 'data';

nets.fine_tuned = load(fullfile(dir, 'b50_e40.mat')); 
nets.fine_tuned = nets.fine_tuned.net;
nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); 
nets.pre_trained = nets.pre_trained.net; 
nets.pre_trained.layers{end}.type = 'softmax';
nets.fine_tuned.layers{end}.type = 'softmax';

data = load(fullfile(expdir, 'imdb-caltech.mat'));

[train_trained, test_trained] = get_svm_data(data, nets.pre_trained);
[train_tuned,  test_tuned] = get_svm_data(data, nets.fine_tuned);

trained_features = [train_trained.features; test_trained.features];
tuned_features = [train_tuned.features; test_tuned.features];
trained_labels = [train_trained.labels; test_trained.labels];
tuned_labels = [train_tuned.labels; test_tuned.labels];

figure1 = figure('Color', 'white'); title('Pretrained'); labels(labels);
tsne_pretrained = tsne(full(trained_features), trained_labels);
saveas(gcf, 'tsne_pretrained.png')

figure2 = figure('Color', 'white'); title('Finetuned'); labels(labels);
tsne_finetuned = tsne(full(tuned_features), tuned_labels);
saveas(gcf, 'tsne_finetuned.png')





