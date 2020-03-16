clc; clf; clear;
matches = load('../data/matches.txt');
uv1 = matches(:,1:2);
uv2 = matches(:,3:4);

I1 = im2double(imread('../data/im1.png'));
I2 = im2double(imread('../data/im2.png'));
K1 = load('../data/K1.txt');
K2 = load('../data/K2.txt');

F = eight_point(uv1, uv2);
E = essential_from_fundamental(F, K1, K2);
Rts = motion_from_essential(E);
[R,t] = choose_solution(uv1, uv2, K1, K2, Rts);
[P1,P2] = camera_matrices(K1, K2, R, t);

% Uncomment for task 4b
% uv1 = load('../data/goodPoints.txt');
% uv2 = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1);

n = size(uv1,1);
X = zeros([n,3]);
for i=1:size(uv1,1)
    X(i,:) = linear_triangulation(uv1(i,:), uv2(i,:), P1, P2);
end

show_point_cloud(X, [-0.6, +0.6], [-0.6, +0.6], [3, 4.2]);
