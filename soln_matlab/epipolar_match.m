function uv2 = epipolar_match(I1, I2, F, uv1)
    % For each point in uv1, finds the matching point in image 2 by
    % an epipolar line search.
    %
    % Args:
    %     I1:  (H x W matrix) Grayscale image 1
    %     I2:  (H x W matrix) Grayscale image 2
    %     F:   (3 x 3 matrix) Fundamental matrix mapping points in image 1 to lines in image 2
    %     uv1: (n x 2 array) Points in image 1
    %
    % Returns:
    %     uv2: (n x 2 array) Best matching points in image 2.
    %
    % Tips:
    % - Image indices must always be integer.
    % - Use round(x) to convert x to an integer.
    % - Use rgb2gray to convert images to grayscale.
    % - Skip points that would result in an invalid access.
    % - Use I(v-w : v+w+1, u-w : u+w) to extract a window of half-width w around (v,u).
    % - Use the sum(..., 'all') function.

    w = 10;
    uv2 = zeros(size(uv1));
    for i=1:size(uv1,1)
        u1 = round(uv1(i,1));
        v1 = round(uv1(i,2));
        if u1 <= w || v1 <= w || u1 >= size(I1,2)-w || v1 >= size(I1,1)-w
            continue
        end

        l = F*[u1 v1 1]';
        W1 = I1(v1-w:v1+w, u1-w:u1+w);

        best_err = inf;
        best_u2 = w;
        for u2=w+1:size(I2,2)-w
            v2 = round(-(l(3) + u2*l(1))/l(2));
            if v2 <= w || v2 >= size(I2,1)-w
                continue
            end
            W2 = I2(v2-w:v2+w, u2-w:u2+w);
            err = sum(abs(W1 - W2), 'all');
            if err < best_err
                best_err = err;
                best_u2 = u2;
            end
        end

        uv2(i,1) = best_u2;
        uv2(i,2) = -(l(3) + best_u2*l(1))/l(2);
    end
end
