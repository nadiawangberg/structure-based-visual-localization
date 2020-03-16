clc; clf; clear;
matches = load('../data/matches.txt');
uv1 = matches(:,1:2);
uv2 = matches(:,3:4);

I1 = im2double(imread('../data/im1.png'));
I2 = im2double(imread('../data/im2.png'));
K1 = load('../data/K1.txt');
K2 = load('../data/K2.txt');

F = eight_point(uv1, uv2);

% rng(0); % Uncomment if you don't want randomized points (use fixed seed)

show_point_matches(I1, I2, uv1, uv2, F);