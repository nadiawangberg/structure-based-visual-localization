function test_normalize_points(pts_n)
    centroid = mean(pts_n);
    dist = mean(vecnorm(pts_n, 2, 2));
    if abs(centroid(1)) > 0.01 || abs(centroid(2)) > 0.01
        fprintf('Translation is NOT GOOD: Centroid was (%g, %g), should be (0, 0).\n', centroid(1), centroid(2));
    else
        fprintf('Translation is GOOD.\n');
    end
    if abs(dist - sqrt(2)) > 0.1
        fprintf('Scaling is NOT GOOD: Mean distance was %g, should be sqrt(2).\n', dist)
    else
        fprintf('Scaling is GOOD.\n');
    end
end