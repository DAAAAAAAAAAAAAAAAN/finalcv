%%
fid = fopen('results_edited.csv', 'rt');
csv = textscan(fid, '%s %s %u %u %u %u %f %f %f %f %f %f', 'HeaderLines', 1, 'Delimiter', ';');

% filter by value for category plots
filter = csv{5} == 200;
for i = 1:12
    %csv{i} = csv{i}(filter);
end

% sort by run time
[s, idx] = sort(csv{12});
for i = 1:12
    csv{i} = csv{i}(idx);
end

y = csv{7};
x = csv{12};

%%
for i = 1:length(x)
    sift_type = csv{2}(i);
    n = csv{5}(i);
    
    % display strictly dominated points a s crosses
    if max(y(1:(i-1))) > y(i)
        % point is strictly dominated
        marker_type = 'x';
        %marker_type = '.'; % hide for color focus plot
    else
        fprintf("Dominant set: %s %s-sift (step size %i), %i vocabulary size, %i training samples: %.3f MAP in %.1fs\n", csv{1}{i}, csv{2}{i}, csv{3}(i), csv{4}(i), csv{5}(i), csv{7}(i), csv{12}(i));
        marker_type = 'o';
    end
    
    if strcmp(sift_type, 'gray')
        color = 'k';
    elseif strcmp(sift_type, 'RGB')
        color = [0.6350, 0.0780, 0.1840];
    elseif strcmp(sift_type, 'rgb')
        color = [0.9290, 0.6940, 0.1250];
    elseif strcmp(sift_type, 'opponent')
        color = 'b';
    end
    
    if ~strcmp(sift_type, 'gray')
        %marker_type = 'd'; % for color focus plot
    end
    
    if n <= 4
        alpha = 0.3;
    elseif n <= 16
        alpha = 0.6;
    elseif n <= 64
        alpha = 0.8;
    elseif n <= 200
        alpha = 1;
    end
    %alpha = 1; % for category plots
    
    % matlab.. sigh
    if marker_type == 'x'
        scatter(x(i), y(i), 50, color,           marker_type, 'MarkerFaceAlpha', alpha, 'LineWidth', 1);
    else
        scatter(x(i), y(i), 50, color, 'filled', marker_type, 'MarkerFaceAlpha', alpha, 'LineWidth', 1);
    end
    hold on;
end
set(gca,'xscale','log')
hold off;
xlabel('runtime (s)');
ylabel('MAP');
%axis([10 4000 0.5 1]) % for some plots