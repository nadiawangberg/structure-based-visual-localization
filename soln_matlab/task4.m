clc; clf; clear;
matches = load('../data/matches.txt');
uv1 = matches(:,1:2);
uv2 = matches(:,3:4);

I1 = im2double(imread('../data/im1.png'));
I2 = im2double(imread('../data/im2.png'));
K1 = load('../data/K1.txt');
K2 = load('../data/K2.txt');

F = eight_point(uv1, uv2);

% rng(0); % Uncomment if you don't want randomized points

% Choose k random points to visualize
k = 8;
sample = randperm(size(uv1, 1), k);
uv1 = uv1(sample,:);
uv2 = uv2(sample,:);
uv2_match = epipolar_match(rgb2gray(I1), rgb2gray(I2), F, uv1);

% Draw points in image 1 and matching point in image 2 (true vs epipolar match)
colors = lines(k);
subplot(121);
imshow(I1);
hold on;
scatter(uv1(:,1), uv1(:,2), 100, colors, 'x', 'LineWidth', 2);
title('Image 1');
subplot(122);
imshow(I2);
hold on;
title('Image 2');
scatter(uv2_match(:,1), uv2_match(:,2), 100, colors, 'o', 'LineWidth', 2);
scatter(uv2(:,1), uv2(:,2), 100, colors, 'x', 'LineWidth', 2);
legend('Found match', 'True match');
