function descriptor = extract_descriptor (image)
    [f, d] = vl_phow(im2single(image), 'Step', 20); % default sift at the moment
    descriptor = d;
end