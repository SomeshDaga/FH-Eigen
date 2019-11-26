function [mmse] = get_landmarks_mmse(features, hr_landmarks, input_landmarks)
%GET_LANDMARKS_MMSE Summary of this function goes here
%   Detailed explanation goes here
hr_features = zeros(length(features),2);
input_features = zeros(length(features),2);

for j=1:length(features)
    hr_features = [hr_landmarks.(features(j)).x,...
                   hr_landmarks.(features(j)).y];
    input_features = [input_landmarks.(features(j)).x,...     
                      input_landmarks.(features(j)).y];
end

mmse = sqrt((hr_features(:,1) - input_features(:,1)).^2 + ...
            (hr_features(:,2) - input_features(:,2)).^2);
end

