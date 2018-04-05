function histogram = extract_histogram (image, C)
    k = size(C, 2);
    d = extract_descriptor(image); % get descriptors
    idx = vl_ikmeanspush(d, C); % assign each descriptor to cluster
    histogram = vl_ikmeanshist(k, idx); % create histogram from assignments
end