function histogram = extract_histogram (image, C, sift_method, sift_type, step_size)
    k = size(C, 2);
    d = extract_descriptor(image, sift_method, sift_type, step_size); % get descriptors
    idx = vl_ikmeanspush(d, C); % assign each descriptor to cluster
    histogram = vl_ikmeanshist(k, idx); % create histogram from assignments
end