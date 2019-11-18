function [patch, original, coords] = Get_SR_Patch(image, features, landmarks, dataset_landmarks, k, padding)
%GET_SR_PATCH Summary of this function goes here
%   Detailed explanation goes here
% Create a matrix of points containing the above features from the input
% image
feature_points = zeros(length(features),2);
for i=1:length(features)
    feature = landmarks.faces.landmark.(features(i));
    feature_points(i,1) = feature.x;
    feature_points(i,2) = feature.y;
end

dissimilarity = zeros(length(dataset_landmarks),1);
for i=1:length(dataset_landmarks)
    candidate_points = zeros(length(features),2);
    for j=1:length(features)
        feature = dataset_landmarks(i).faces.landmark.(features(j));
        candidate_points(j,1) = feature.x;
        candidate_points(j,2) = feature.y;
    end
    
    % Calculate similarity with mouth features of the input image
    % The higher the similarity, the better the fit
    % We disable scaling factors because the images are aligned and
    % and have been set to similar scales already via setting the center
    % eye distance to a fixed number of pixels
    dissimilarity(i) = procrustes(feature_points, candidate_points, 'scaling', false);
end

% Choose the k highest similarities
[vals, idxs] = mink(dissimilarity, k);

% Re-create this region using just the images that are most similar

% Firstly, for all of the candidates images and the input image, we extract
% the pixels at the mouth region
% We need the extracted regions to have the same dimensions for the math to
% work out. Hence, we need to find the most conservative size of bounding
% box that captures the mouth region in all the images
sr_x_max = max(feature_points(:,1)) + padding;
sr_x_min = min(feature_points(:,1)) - padding;
sr_y_max = max(feature_points(:,2)) + padding;
sr_y_min = min(feature_points(:,2)) - padding;

% Create a 3D array to store the mouth points for all candidate images
candidate_points = zeros(length(features),2,length(idxs));
for i=1:length(idxs)
    for j=1:length(features)
        feature = dataset_landmarks(idxs(i)).faces.landmark.(features(j));
        candidate_points(j,:,i) = [feature.x feature.y];
    end
end

candidate_bboxes = zeros(length(idxs)+1,2);
bbox_centers = zeros(length(idxs)+1,2);
for i=1:length(idxs)
    x_max = max(candidate_points(:,1,i)) + padding;
    x_min = min(candidate_points(:,1,i)) - padding;
    y_max = max(candidate_points(:,2,i)) + padding;
    y_min = min(candidate_points(:,2,i)) - padding;
    bbox_centers(i,:) = [(x_max + x_min)/2 (y_max + y_min)/2];
    candidate_bboxes(i,:) = [(x_max - x_min) (y_max - y_min)];
end
candidate_bboxes(end,:) = [(sr_x_max - sr_x_min) (sr_y_max - sr_y_min)];
bbox_centers(end,:) = [(sr_x_max + sr_x_min)/2 (sr_y_max + sr_y_min)/2];
best_bbox_size = [max(candidate_bboxes(:,1)) max(candidate_bboxes(:,2))];

% Combine the FFTs from the candidate images based on their 
norm = sum(vals);
final_fft = zeros(best_bbox_size(1),best_bbox_size(2));
[original, coords] = get_region(image, best_bbox_size, bbox_centers(end,:));

for i=1:length(idxs)
    % Get the equivalent face region from each of our best candidates
    region = get_region(imread(dataset_landmarks(idxs(i)).file),best_bbox_size,bbox_centers(i,:));
    if i==1
        final_fft = (vals(i)/norm)*fft2(region);
    else
        temp_fft = (vals(i)/norm)*fft2(region);
        final_fft = temp_fft + final_fft;
    end
end

patch = uint8(real(ifft2(final_fft)));

end

