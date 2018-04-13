%% Daan Smedinga 10560963 and Jens Dudink 11421479
run('C:\Users\Daan\Downloads\vlfeat-0.9.21\toolbox\vl_setup')
path(path, 'C:\Sync\school\s2b4 CV1\finalcv\liblinear-2.20\windows')

%% looking at descriptor
I = im2single(imread('cameraman.tif'));
Is = vl_imsmooth(I, sqrt((1)^2 - .25)) ;
[f1,d1] = vl_dsift(Is, 'Size', 3);
f1(3,:) = 1;
f1(4,:) = 0;
[f2,d2] = vl_sift(I, 'frames', f1(:,1), 'Magnif', 3);
disp("frames are exactly the same:");
cat(2, f1(:,1), f2(:,1))
disp("but descriptors are very different:");
cat(2, d1(1:10,1), d2(1:10,1))

%% looking at descriptors in detail
x = 1000; % selecting a sift frame with weak gradient
%x = 24000; % selecting a sift frame with strong gradient
img = imread('cameraman.tif');
I = im2single(img);
binSize = 3;
magnif = 3;
Is = vl_imsmooth(I, sqrt((binSize/magnif)^2 - .25)) ;

[f1, d1] = vl_dsift(Is, 'size', binSize) ;
f1(3,:) = binSize/magnif;
f1(4,:) = 0;
[f2, d2] = vl_sift(I, 'frames', f1(:,x));
disp("frames are exactly the same:");
cat(2, f1(:,x), f2(:,1))
disp("descriptors (strong gradient: not) very different:");
cat(2, d1(1:10,x), d2(1:10,1))

imshow(img);
h3 = vl_plotsiftdescriptor(d1(:,x),f1(:,x)) ;
h4 = vl_plotsiftdescriptor(d2(:,1),f2(:,1)) ;
set(h3,'color','g', 'LineWidth', 3) ;
set(h4,'color','r', 'LineWidth', 1.5) ;