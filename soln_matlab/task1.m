clc; clf; clear;
matches = load('../data/matches.txt');
pts = matches(:,1:2);

[pts_n,T] = normalize_points(pts);

fprintf('Checking that the points satisfy the normalization criteria...\n');
test_normalize_points(pts_n);

fprintf('Checking that the transformation matrix performs the same operation...\n');
pts_n = (T*[pts' ; ones(1,size(pts,1))])';
pts_n = pts_n(:,1:2) ./ pts_n(:,3);
test_normalize_points(pts_n);

