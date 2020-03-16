function show_point_matches(I1, I2, uv1, uv2, F)
    % Plots k randomly chosen matching point pairs in image 1 and
    % image 2. If the fundamental matrix F is given, it also plots the
    % epipolar lines.

    k = 8;
    sample = randperm(size(uv1, 1), k);
    uv1 = uv1(sample,:);
    uv2 = uv2(sample,:);

    colors = lines(k);
    subplot(121);
    imshow(I1);
    hold on;
    scatter(uv1(:,1), uv1(:,2), 100, colors, 'x', 'LineWidth', 2);
    title('Image 1');
    subplot(122);
    imshow(I2);
    hold on;
    scatter(uv2(:,1), uv2(:,2), 100, colors, 'o', 'LineWidth', 2);
    title('Image 2');

    if exist('F', 'var')
        for i=1:k
            l = F*[uv1(i,:) 1]';
            draw_line(l, colors(i,:))
        end
    end
end

function draw_line(l, color)
    % Draws the line satisfies the line equation
    %     x l[0] + y l[1] + l[2] = 0
    % clipped to the current plot's box (xlim, ylim).

    ulim = xlim();
    vlim = ylim();
    c = l(1);
    s = l(2);
    rho = -l(3);
    if abs(s) > abs(c)
        u1 = ulim(1);
        u2 = ulim(2);
        v1 = (rho-u1*c)/s;
        v2 = (rho-u2*c)/s;
        [v1,u1] = clamp_line_parameters(v1, u1, vlim(1), vlim(2), rho, s, c);
        [v2,u2] = clamp_line_parameters(v2, u2, vlim(1), vlim(2), rho, s, c);
    else
        v1 = vlim(1);
        v2 = vlim(2);
        u1 = (rho-v1*s)/c;
        u2 = (rho-v2*s)/c;
        [u1,v1] = clamp_line_parameters(u1, v1, ulim(1), ulim(2), rho, c, s);
        [u2,v2] = clamp_line_parameters(u2, v2, ulim(1), ulim(2), rho, c, s);
    end
    plot([u1, u2], [v1, v2], 'color', color, 'linewidth', 2);
end

function [a, b] = clamp_line_parameters(a, b, a_min, a_max, rho, A, B)
    if a < a_min || a > a_max
        a = max(a_min, min(a_max, a));
        b = (rho - a*A)/B;
    end
end
