function X = linear_triangulation(uv1, uv2, P1, P2)
    % Compute the 3D position of a point from 2D correspondences.
    %
    % Args:
    %    uv1:    2D projection of the point in image 1
    %    uv2:    2D projection of the point in image 2
    %    P1:     Projection matrix with shape 3 x 4 for image 1
    %    P2:     Projection matrix with shape 3 x 4 for image 2
    %
    % Returns:
    %    X:      3D coordinates of point in the camera frame of image 1
    %            (not homogeneous!)
    %
    % See HZ Ch. 12.2: Linear triangulation methods (p312)
    
    A = zeros(4);
    A(1,:) = uv1(1)*P1(3,:) - P1(1,:);
    A(2,:) = uv1(2)*P1(3,:) - P1(2,:);
    A(3,:) = uv2(1)*P2(3,:) - P2(1,:);
    A(4,:) = uv2(2)*P2(3,:) - P2(2,:);
    [~,~,V] = svd(A);
    X_tilde = V(:,4);
    X = X_tilde(1:3)/X_tilde(4);
end