function [patch, original, coords] = Get_SR_Patch(image, features, landmarks, dataset_landmarks, center_feature, k, padding)
%GET_SR_PATCH Summary of this function goes here
%   Detailed explanation goes here
% Create a matrix of points containing the above features from the input
% image
[im_h, im_w] = size(image);
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
    [dissimilarity(i), ~, transform] = procrustes(feature_points, candidate_points, 'scaling', false, 'reflection', false);
    if i==1
        transforms = repmat(struct(transform),length(dataset_landmarks),1);
    end
    transforms(i) = transform;
end

% Choose the k highest similarities (or lowest dissimlarities)
[vals, idxs] = mink(dissimilarity, k);
transforms = transforms(idxs);

% Re-create this region using just the images that are most similar

% Firstly, for all of the candidates images and the input image, we extract
% the pixels at the mouth region
% We need the extracted regions to have the same dimensions for the math to
% work out. Hence, we need to find the most conservative size of bounding
% box that captures the mouth region in all the images


% Create a 3D array to store the mouth points for all candidate images
candidate_points = zeros(length(features),2,length(idxs));
for i=1:length(idxs)
    for j=1:length(features)
        feature = dataset_landmarks(idxs(i)).faces.landmark.(features(j));
        candidate_points(j,:,i) = [feature.x feature.y];
    end
end

candidate_bboxes = zeros(length(idxs)+1,4);
bbox_centers = zeros(length(idxs)+1,2);
for i=1:length(idxs)
    bbox_centers(i,:) = [dataset_landmarks(idxs(i)).faces.landmark.(center_feature).x ...
                         dataset_landmarks(idxs(i)).faces.landmark.(center_feature).y];
    % Find the max deviations of features from the given centers
    % in the +ve x and y directions
    max_x_dist = max(candidate_points(:,1,i) - bbox_centers(i,1)) + padding;
    max_y_dist = max(candidate_points(:,2,i) - bbox_centers(i,2)) + padding;
    min_x_dist = abs(min(candidate_points(:,1,i) - bbox_centers(i,1))) + padding;
    min_y_dist = abs(min(candidate_points(:,2,i) - bbox_centers(i,2))) + padding;
    candidate_bboxes(i,:)= [min_x_dist max_x_dist min_y_dist max_y_dist];
end

bbox_centers(end,:) = [landmarks.faces.landmark.(center_feature).x ...
                       landmarks.faces.landmark.(center_feature).y];
max_x_dist = max(feature_points(:,1) - bbox_centers(end,1)) + padding;
max_y_dist = max(feature_points(:,2) - bbox_centers(end,2)) + padding;
min_x_dist = abs(min(feature_points(:,1) - bbox_centers(end,1))) + padding;
min_y_dist = abs(min(feature_points(:,2) - bbox_centers(end,2))) + padding;

candidate_bboxes(end,:)= [min_x_dist max_x_dist min_y_dist max_y_dist];

best_bbox_size = [min(candidate_bboxes(:,1)) max(candidate_bboxes(:,2)) ...
                  min(candidate_bboxes(:,3)) max(candidate_bboxes(:,4))];

% Combine the FFTs from the candidate images based on their 
norm = sum(vals);
[original, coords] = get_region(image, best_bbox_size, bbox_centers(end,:));

% Idea 1: Just replace the patch with a normalized linear combination of
% the best K patches based on similarity value alone
% final_fft = zeros(coords(2)-coords(1), coords(4)-coords(3));
% 
% for i=1:length(idxs)
%     region = get_region(imread(dataset_landmarks(idxs(i)).file), best_bbox_size, bbox_centers(i,:));
%     if i==1
%         final_fft = (vals(i)/norm)*fft2(region);
%     else
%         temp_fft = (vals(i)/norm)*fft2(region);
%         final_fft = temp_fft + final_fft;
%     end
% end
% 
% patch = uint8(real(ifft2(final_fft)));

% Idea 2: Perform a PCA-based procedure on the K most similar regions
% and replace the patch
original = reshape(original, [numel(original), 1]);
for i=1:length(idxs)
    train_region = get_region(imread(dataset_landmarks(idxs(i)).file), best_bbox_size, bbox_centers(i,:));
%     subplot(1,1,1);
%     imshow(train_region);
%     waitforbuttonpress
    Y(:,i) = reshape(train_region,[numel(train_region),1]);
end
Y = double(Y);
mY = mean(Y, 2);
Y = Y - repmat(mY, [1, length(idxs)]);
[V, Dl] = eig( Y'*Y );
% Dl = Dl(end-k:end,end-k:end);
% V = V(:,end-k:end);
El = Y * V * Dl^(-0.5);
Vl = V * Dl^(-0.5);
weights = El' * (double(original) - mY);
patch = El * weights + mY;
patch = uint8(reshape(patch, [coords(4)-coords(3) + 1, coords(2) - coords(1) + 1]));
end

