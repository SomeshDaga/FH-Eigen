function [X,Y] = get_feature_points(features, landmarks, center_flag, center_feature)
%GET_FEATURE_POINTS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    center_flag = false;
end

x_center = 0;
y_center = 0;
if center_flag
    x_center = landmarks.faces.landmark.(center_feature).x;
    y_center = landmarks.faces.landmark.(center_feature).y;
end

% Store the x,y positions of the given features
j = 0;
for i=1:length(features)
    feature = landmarks.faces.landmark.(features(i));
    if (center_flag && center_feature ~= features(i)) || ~center_flag
        j = j+1;
        X(j) = feature.x - x_center;
        Y(j) = feature.y - y_center;
    end
end

end

