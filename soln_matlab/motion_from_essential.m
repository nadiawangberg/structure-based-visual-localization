function [Rts] = motion_from_essential(E)
    % Computes the four possible decompositions of E into
    % a relative rotation and translation.
    % See HZ Ch. 9.7 (p259): Result 9.19

    [U,~,V] = svd(E);

    % Make sure we return rotation matrices with det(R) == 1
    if det(U) < 0
        U = -U;
    end
    if det(V) < 0
        V = -V;
    end

    W = [0 -1 0 ; +1 0 0 ; 0 0 1];
    R1 = U*W*V';
    R2 = U*W'*V';
    t1 = U(:,3);
    t2 = -U(:,3);
    Rts = zeros(3,4,4);
    Rts(:,:,1) = [R1 t1];
    Rts(:,:,2) = [R1 t2];
    Rts(:,:,3) = [R2 t1];
    Rts(:,:,4) = [R2 t2];
end