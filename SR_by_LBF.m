function [ ] = SR_by_LBF(par, sr_image, bicubic_image, landmarks, dataset_landmarks)
%SR_BY_LBF Summary of this function goes here
%   Detailed explanation goes here

% Find the images and regions from the datasets that best match the facial
% landmarks for a specific facial component of the input image

% 1) Mouth Region
mouth_features = ["mouth_left_corner",...
                  "mouth_upper_lip_left_contour1",...
                  "mouth_upper_lip_left_contour2",...
                  "mouth_upper_lip_left_contour3",...
                  "mouth_upper_lip_top",...
                  "mouth_upper_lip_bottom",...
                  "mouth_upper_lip_right_contour3",...
                  "mouth_upper_lip_right_contour2",...
                  "mouth_upper_lip_right_contour1",...
                  "mouth_right_corner",...
                  "mouth_lower_lip_right_contour1",...
                  "mouth_lower_lip_right_contour2",...
                  "mouth_lower_lip_right_contour3",...
                  "mouth_lower_lip_top",...
                  "mouth_lower_lip_bottom",...
                  "mouth_lower_lip_left_contour3",...
                  "mouth_lower_lip_left_contour2",...
                  "mouth_lower_lip_left_contour1"];
% Create a matrix of points containing the above features from the input
% image
sr_mouth_points = zeros(length(mouth_features),2);
for i=1:length(mouth_features)
    feature = landmarks.faces.landmark.(mouth_features(i));
    sr_mouth_points(i,1) = feature.x;
    sr_mouth_points(i,2) = feature.y;
end

dissimilarity = zeros(length(dataset_landmarks),1);
for i=1:length(dataset_landmarks)
    candidate_mouth_points = zeros(length(mouth_features),2);
    for j=1:length(mouth_features)
        feature = dataset_landmarks(i).faces.landmark.(mouth_features(j));
        candidate_mouth_points(j,1) = feature.x;
        candidate_mouth_points(j,2) = feature.y;
    end
    
    % Calculate similarity with mouth features of the input image
    % The higher the similarity, the better the fit
    % We disable scaling factors because the images are aligned and
    % and have been set to similar scales already via setting the center
    % eye distance to a fixed number of pixels
    dissimilarity(i) = procrustes(sr_mouth_points, candidate_mouth_points, 'scaling', false);
end

% Choose the k highest similarities
k = 2;
[vals, idxs] = mink(dissimilarity, k);

% Re-create this region using just the images that are most similar

% Firstly, for all of the candidates images and the input image, we extract
% the pixels at the mouth region
% We need the extracted regions to have the same dimensions for the math to
% work out. Hence, we need to find the most conservative size of bounding
% box that captures the mouth region in all the images
padding = 5;
sr_x_max = max(sr_mouth_points(:,1)) + padding;
sr_x_min = min(sr_mouth_points(:,1)) - padding;
sr_y_max = max(sr_mouth_points(:,2)) + padding;
sr_y_min = min(sr_mouth_points(:,2)) - padding;

% Create a 3D array to store the mouth points for all candidate images
candidate_mouth_points = zeros(length(mouth_features),2,length(idxs));
for i=1:length(idxs)
    for j=1:length(mouth_features)
        feature = dataset_landmarks(idxs(i)).faces.landmark.(mouth_features(j));
        candidate_mouth_points(j,:,i) = [feature.x feature.y];
    end
end

candidate_bboxes = zeros(length(idxs)+1,2);
bbox_centers = zeros(length(idxs)+1,2);
for i=1:length(idxs)
    x_max = max(candidate_mouth_points(:,1,i)) + padding;
    x_min = min(candidate_mouth_points(:,1,i)) - padding;
    y_max = max(candidate_mouth_points(:,2,i)) + padding;
    y_min = min(candidate_mouth_points(:,2,i)) - padding;
    bbox_centers(i,:) = [(x_max + x_min)/2 (y_max + y_min)/2];
    candidate_bboxes(i,:) = [(x_max - x_min) (y_max - y_min)];
end
candidate_bboxes(end,:) = [(sr_x_max - sr_x_min) (sr_y_max - sr_y_min)];
bbox_centers(end,:) = [(sr_x_max + sr_x_min)/2 (sr_y_max + sr_y_min)/2];
best_bbox_size = [max(candidate_bboxes(:,1)) max(candidate_bboxes(:,2))];

% Show the input image's extracted region
imshow(get_region(bicubic_image, best_bbox_size, bbox_centers(end,:)));
waitforbuttonpress
% imagesc(abs(fftshift(fft2(get_region(bicubic_image, best_bbox_size, bbox_centers(end,:))))));
% waitforbuttonpress
% imagesc(abs(fftshift(fft2(get_region(sr_image, best_bbox_size, bbox_centers(end,:))))));
% waitforbuttonpress

% Combine the FFTs from the candidate images based on their 
norm = sum(vals);
final_fft = zeros(best_bbox_size(1),best_bbox_size(2));
for i=1:length(idxs)
    if i==1
        final_fft = (vals(i)/norm)*fft2(get_region(imread(dataset_landmarks(idxs(i)).file),best_bbox_size,bbox_centers(i,:)));
        size(final_fft);
    else
        temp_fft = (vals(i)/norm)*fft2(get_region(imread(dataset_landmarks(idxs(i)).file),best_bbox_size,bbox_centers(i,:)));
        size(temp_fft);
        final_fft = temp_fft + final_fft;
    end
%     imagesc(abs(fftshift(fft2(get_region(imread(dataset_landmarks(idxs(i)).file),best_bbox_size,bbox_centers(i,:))))));
%     waitforbuttonpress
end
[sr_mouth, coords] = get_region(sr_image, best_bbox_size, bbox_centers(end,:));
imshow(sr_mouth);
waitforbuttonpress
imshow(uint8(real(ifft2(final_fft))));
waitforbuttonpress
new_sr_image = sr_image;
new_sr_image(coords(3):coords(4),coords(1):coords(2)) = uint8(real(ifft2(final_fft)));
% imshow(histeq(new_sr_image));
imshow(new_sr_image);
waitforbuttonpress
end

