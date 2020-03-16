function show_point_cloud(X, my_xlim, my_ylim, my_zlim)
    % Creates a mouse-controllable 3D plot of the input points.
    
    % This could be changed to use scatter if you want to
    % provide a per-point color. Otherwise, the plot function
    % is much faster.
    plot3(X(:,1), X(:,3), X(:,2), '.');
    
    grid on;
    box on;
    axis equal;
    axis vis3d;
    camproj perspective;
    ylim(my_zlim);
    xlim(my_xlim);
    zlim(my_ylim);
    set(gca, 'ZDir', 'reverse');
    xlabel('X');
    ylabel('Z');
    zlabel('Y');
end