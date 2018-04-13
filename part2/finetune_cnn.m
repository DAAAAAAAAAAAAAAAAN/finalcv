function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run(fullfile(fileparts(mfilename('fullpath')), ...
  'matconvnet', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.train.gpus = [];



%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getCaltechIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
splits = {'train', 'test'};

%% TODO: Implement your loop here, to create the data structure described in the assignment

total = 0;
n_classes = length(classes);
n_splits = length(splits);

for s = 1:n_splits
    % 80% of the total number of images belongs to the training set, the
    % rest belongs to the test set
    if strcmp(splits(s), splits(2))
        n_imgs = 50;
    else
        n_imgs = 400;
    end
    
    for i = 1:n_classes
        file_name = char(strcat('../Caltech4/ImageData/', classes(i),'_', splits(s), '/'));
      
        
        for j = 1:n_imgs
            image_name = strcat(file_name,'img',num2str(j,'%.3d'),'.jpg');
            image = imread(image_name);
            
            % Count total number of images
            if size(image,3) == 3 
                total = total + 1; 
            end
        end
    end
end

sets = single(zeros(1, total));
labels = single(zeros(1, total));
data = single(zeros(32, 32, 3, total));

index = 1;
for s = 1:n_splits
    if strcmp(splits(s), splits(1))
        n_imgs = 400;
    else
        n_imgs = 50;
    end
    
    for i = 1:n_classes
        file_name = char(strcat('../Caltech4/ImageData/', classes(i),'_', splits(s), '/'));
        
        for j = 1:n_imgs
            path = strcat(file_name, 'img', num2str(j,'%.3d'), '.jpg');
            image = imread(path);
            
            if size(image, 3) == 3
                image = single(im2double(image));
                
                % Resize image
                image = imresize(image, [32 32]); 
                data(:, :, :, index) = image;
                
                % Split training and test
                if strcmp(splits(s), splits(1)) % setting split
                    sets(index) = 1;
                else
                    sets(index) = 2;
                end
                
                % Add to specific class
                if strcmp(classes(i), classes(1)) 
                    labels(index) = 1;
                elseif strcmp(classes(i), classes(2))
                    labels(index) = 2;
                elseif strcmp(classes(i), classes(3))
                    labels(index) = 3;
                elseif strcmp(classes(i), classes(4))
                    labels(index) = 4;
                end
                index = index + 1;
            end
        end
    end
end

sets = single(sets);
labels = single(labels);
data = single(data);
%%
% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);
data = bsxfun(@minus, data, dataMean);

imdb.images.data = data ;
imdb.images.labels = single(labels) ;
imdb.images.set = sets;
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end
