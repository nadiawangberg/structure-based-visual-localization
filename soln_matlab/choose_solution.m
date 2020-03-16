function [R,t] = choose_solution(uv1, uv2, K1, K2, Rts)
    % Chooses among the rotation and translation solutions Rts
    % the one which gives the most points in front of both cameras.

    num_visible = zeros(1,size(Rts,3));
    for i=1:size(Rts, 3)
        R = Rts(1:3,1:3,i);
        t = Rts(1:3,4,i);
        [P1,P2] = camera_matrices(K1, K2, R, t);
        for j=1:size(uv1,1)
            X1 = linear_triangulation(uv1(j,:), uv2(j,:), P1, P2);
            X2 = R*X1 + t;
            if X1(3) > 0 && X2(3) > 0
                num_visible(i) = num_visible(i) + 1;
            end
        end
    end

    [N,i] = max(num_visible);
    fprintf('Choosing solution %d (%d points visible).\n', i, N);
    R = Rts(1:3,1:3,i);
    t = Rts(1:3,4,i);
end
