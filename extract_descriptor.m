function descriptors = extract_descriptor (image, sift_method, sift_type, step_size)
    bin_size = 3;
    magnif = 3;
    
    if size(image, 3) > 1
        image_gray = rgb2gray(image);
    else
        image_gray = image;
    end
    
    if strcmp(sift_method,'keypoint')
        [frames, descriptors] = vl_sift(im2single(image_gray));
    elseif strcmp(sift_method, 'dense')
        % doc says to blur first, but it always hurts performance it seems
        %image_gray = vl_imsmooth(image_gray, sqrt((bin_size/magnif)^2 - .25));
        [frames, descriptors] = vl_dsift(im2single(image_gray), 'size', bin_size, 'Step', step_size);
        frames(3,:) = 1;
        frames(4,:) = 0;
    end
    
    if ~strcmp(sift_type, 'gray') && size(image,3) == 1
        % some images are grayscale by themselves
        % just duplicate the layers
        image = repmat(image, 1, 1, 3);
    end
    
    if strcmp(sift_type, 'gray')
        % already computed
        return
    elseif strcmp(sift_type, 'RGB')
        I = image;
    elseif strcmp(sift_type, 'rgb')
        I = double(image) ./ sum(image,3);
    elseif strcmp(sift_type, 'opponent')
        I = zeros(size(image));
        R = image(:,:,1);
        G = image(:,:,2);
        B = image(:,:,3);
        I(:,:,1) = (R-G)/sqrt(2);
        I(:,:,2) = (R+G-2*B)/sqrt(6);
        I(:,:,3) = (R+G+B)/sqrt(3);
    end
    
    layer_count = size(I,3);
    descriptors_per_layer = cell(1, layer_count);
    for i = 1:layer_count
        [f,d] = vl_sift(im2single(I(:,:,i)), 'Magnif', magnif, 'frames', frames);
        descriptors_per_layer{i} = d;
    end
    descriptors = cat(1, descriptors_per_layer{:});
end