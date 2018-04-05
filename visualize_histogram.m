function visualize_histogram (image, h)
    figure(2);
    subplot(1,2,1);
    histogram(h, length(h));
    subplot(1,2,2);
    imshow(image);
end