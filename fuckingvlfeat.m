
%%
run_experiment(sift_methods{1}, sift_types{1}, step_sizes(1), vocabulary_sizes(3), numbers_of_training_samples(3), number_of_testing_samples, false);

%%
I = im2single(imread('cameraman.tif'));
Is = vl_imsmooth(I, sqrt((1)^2 - .25)) ;
[f1,d1] = vl_dsift(Is, 'Size', 3);
f1(3,:) = 1;
f1(4,:) = 0;
[f2,d2] = vl_sift(I, 'frames', f1(:,1), 'Magnif', 3);
f1(:,1)
f2(:,1)
d1(1:10,1)
d2(1:10,1)
% sum(sum(abs(d1-d2)))

%% vl_feat doesn't work as advertised
x = 24000; % selecting a sift frame
img = imread('cameraman.tif');
I = im2single(img);
binSize = 3;
magnif = 3;
Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

[f1, d1] = vl_dsift(Is, 'size', binSize) ;
f1(3,:) = binSize/magnif;
f1(4,:) = 0;
[f2, d2] = vl_sift(I, 'frames', f1(:,x));
f1(:,x)
f2(:,1)
d1(1:10,x)
d2(1:10,1)

imshow(img);
h3 = vl_plotsiftdescriptor(d1(:,x),f1(:,x)) ;
h4 = vl_plotsiftdescriptor(d2(:,1),f2(:,1)) ;
set(h3,'color','g', 'LineWidth', 2) ;
set(h4,'color','r') ;