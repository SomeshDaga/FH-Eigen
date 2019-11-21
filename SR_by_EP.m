function [] = SR_by_EP(par, sr, bicubic, landmarks, dataset_landmarks)
%SR_BY_EP Summary of this function goes here
%   Detailed explanation goes here

% Step 1) Choose a region/patch to resolve
features = ["mouth_left_corner",...
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
bbox_sizes = zeros(length(dataset_landmarks) + 1, 2);
bbox_centers = zeros(length(dataset_landmarks) + 1, 2);
padding = 10;
feature_pts = zeros(length(features),2);
for i=1:length(features)
    feature_pts(i,1) = landmarks.faces.landmark.(features(i)).x;
    feature_pts(i,2) = landmarks.faces.landmark.(features(i)).y;
end
bbox_sizes(end, :) = [max(feature_pts(:,1)) - min(feature_pts(:,1)) + padding, ...
                      max(feature_pts(:,2)) - min(feature_pts(:,2)) + padding];
bbox_centers(end, :) = [(max(feature_pts(:,1)) + min(feature_pts(:,1))) / 2, ...
                        (max(feature_pts(:,2)) + min(feature_pts(:,2))) / 2];
for i=1:length(dataset_landmarks)
    feature_pts = zeros(length(features),2);
    for j=1:length(features)
        feature_pts(j,1) = dataset_landmarks(i).faces.landmark.(features(j)).x;
        feature_pts(j,2) = dataset_landmarks(i).faces.landmark.(features(j)).y;
    end
    bbox_sizes(i,:) = [max(feature_pts(:,1)) - min(feature_pts(:,1)) + padding, ...
                       max(feature_pts(:,2)) - min(feature_pts(:,2)) + padding];
    bbox_centers(i,:) = [(max(feature_pts(:,1)) + min(feature_pts(:,1))) / 2, ...
                         (max(feature_pts(:,2)) + min(feature_pts(:,2))) / 2];
end

% Find the best bbox size
best_bbox_size = [max(bbox_sizes(:,1)) max(bbox_sizes(:,2))];

% Create vector to store the ssim between the same regions between the
% bicubic images and the training images
ssims = zeros(length(dataset_landmarks), 1);

% Create the L matrix from all the training images with the given patch
% extracted using the best_bbox_size
L = zeros((best_bbox_size(1))* (best_bbox_size(2)), length(dataset_landmarks));
for i=length(dataset_landmarks)+1:-1:1
    if i < length(dataset_landmarks) + 1
        train_img_hr = imread(dataset_landmarks(i).file);
        [height, width, ~] = size(train_img_hr);
    else
        [height, width] = size(bicubic);
    end
    
    x_top = uint8(bbox_centers(i,1) + best_bbox_size(1)/2);
    x_bottom = x_top + 1 - best_bbox_size(1);
    y_right = uint8(bbox_centers(i,2) + best_bbox_size(2)/2);
    y_left = y_right + 1 - best_bbox_size(2);

    if x_top > height
        x_top = height;
        x_bottom = (x_top + 1) - best_bbox_size(1);
    elseif x_bottom < 1
        x_bottom = 1;
        x_top = x_bottom - 1 + best_bbox_size(1);
    end
    
    if y_right > width
        y_right = width;
        y_left = (y_right + 1) - best_bbox_size(2);
    elseif y_left < 1
        y_left = 1;
        y_right = y_left - 1 + best_bbox_size(2);
    end

    if i < length(dataset_landmarks) + 1
        region = train_img_hr(y_left:y_right ,x_bottom:x_top);
        ssims(i) = ssim(region, test_region_img );
        L(:,i) = reshape(region, [(best_bbox_size(1)) * (best_bbox_size(2)),1]);
    else
        test_region_img = bicubic(y_left:y_right, x_bottom:x_top);
        test_region = reshape(test_region_img, [(best_bbox_size(1)) * (best_bbox_size(2)),1]);
    end
%     imshow(uint8(region));
%     waitforbuttonpress
end
% Extract the feature region from the bicubic interpolated image
k = 50;
mean_L = mean(L,2);
L = L - repmat(mean_L, [1 length(dataset_landmarks)]);
[V, D] = eig(L'*L);
D = D(end-k:end,end-k:end);
V = V(:,end-k:end);
E_h = L*V*D^(-0.5);
w = E_h' * (double(test_region) - mean_L);
% c = V*D^(-0.5)*w;
% 
% x = mean_L;
% for i=1:length(c)
%     x = x + c(i)*L(:,i);
% end
for j = 1:size(w, 1)
    if abs(w(j)) > par.alpha*sqrt(D(j,j))
        w(j) = sign(w(j))*par.alpha*sqrt(D(j,j));
    end
end
x = real(E_h * w + mean_L);
x = reshape(x, [best_bbox_size(2), best_bbox_size(1)]);

sr_new = sr;
sr_new(y_left:y_right,x_bottom:x_top) = x;
imshow(uint8(sr_new));
waitforbuttonpress

% Consider using the blockproc function to partition image into consistent
% patches if required

% Get the width and length of the patch/bounding box required to capture
% this facial feature in the input image

% Step 2) Find the equivalent regions/patches in the high-resolution images

% Step 3) Find eigen-regions and select the best K eigen regions to perform
%         the eigen-based super resolution of that region
end

