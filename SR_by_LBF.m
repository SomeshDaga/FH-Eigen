function [ patched_image ] = SR_by_LBF(par, input, hr, sr_image, bicubic_image, landmarks_sr, landmarks_bicubic, dataset_landmarks, mean_hr)
%SR_BY_LBF Summary of this function goes here
%   Detailed explanation goes here

patched_image = sr_image;

% Find the images and regions from the datasets that best match the facial
% landmarks for a specific facial component of the input image

% 0) Face Contour
face_features = ["contour_left9",...
                 "contour_left8",...
                 "contour_left7",...
                 "contour_left6",...
                 "contour_left5",...
                 "contour_left4",...
                 "contour_left3",...
                 "contour_left2",...
                 "contour_left1",...
                 "contour_chin",...
                 "contour_right1",...
                 "contour_right2",...
                 "contour_right3",...
                 "contour_right4",...
                 "contour_right5",...
                 "contour_right6",...
                 "contour_right7",...
                 "contour_right8",...
                 "contour_right9"];

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
% 2) Left Eye Region
left_eye_features = ["left_eye_left_corner",...
                     "left_eye_upper_left_quarter",...
                     "left_eye_top",...
                     "left_eye_upper_right_quarter",...
                     "left_eye_right_corner",...
                     "left_eye_lower_right_quarter",...
                     "left_eye_bottom",...
                     "left_eye_lower_left_quarter",...
                     "left_eye_center",...
                     "left_eye_pupil"];
left_eyebrow_features = ["left_eyebrow_left_corner",...
                         "left_eyebrow_upper_left_quarter",...
                         "left_eyebrow_upper_middle",...
                         "left_eyebrow_upper_right_quarter",...
                         "left_eyebrow_right_corner",...
                         "left_eyebrow_lower_right_quarter",...
                         "left_eyebrow_lower_middle",...
                         "left_eyebrow_lower_left_quarter"];
% 3) Right Eye Region
right_eye_features = ["right_eye_left_corner",...
                      "right_eye_upper_left_quarter",...
                      "right_eye_top",...
                      "right_eye_upper_right_quarter",...
                      "right_eye_right_corner",...
                      "right_eye_lower_right_quarter",...
                      "right_eye_bottom",...
                      "right_eye_lower_left_quarter",...
                      "right_eye_center",...
                      "right_eye_pupil"];
right_eyebrow_features = ["right_eyebrow_left_corner",...
                          "right_eyebrow_upper_left_quarter",...
                          "right_eyebrow_upper_middle",...
                          "right_eyebrow_upper_right_quarter",...
                          "right_eyebrow_right_corner",...
                          "right_eyebrow_lower_right_quarter",...
                          "right_eyebrow_lower_middle",...
                          "right_eyebrow_lower_left_quarter"];

% 4) Nose Region
nose_features = ["nose_contour_left3",...
                 "nose_contour_left2",...
                 "nose_contour_left1",...
                 "nose_left",...
                 "nose_contour_lower_middle",...
                 "nose_right",...
                 "nose_contour_right1",...
                 "nose_contour_right2",...
                 "nose_contour_right3",...
                 "nose_tip"];

[mouth_patch, ~, coords] = Get_SR_Patch(bicubic_image, input, mouth_features, landmarks_bicubic, dataset_landmarks, "mouth_lower_lip_top", 12, 8);
mouth_patch = imhistmatch(mouth_patch, patched_image(coords(3):coords(4),coords(1):coords(2)), 'method', 'polynomial');
patched_image(coords(3):coords(4),coords(1):coords(2)) = mouth_patch;
sr_mouth = sr_image(coords(3):coords(4),coords(1):coords(2));
bicubic_mouth = bicubic_image(coords(3):coords(4),coords(1):coords(2));
hr_mouth = hr(coords(3):coords(4),coords(1):coords(2));

[nose_patch, ~, coords] = Get_SR_Patch(sr_image, input, nose_features, landmarks_sr, dataset_landmarks, "nose_tip", 10, 3);
nose_patch = imhistmatch(nose_patch, patched_image(coords(3):coords(4),coords(1):coords(2)), 'method', 'polynomial');
patched_image(coords(3):coords(4),coords(1):coords(2)) = nose_patch;

% left_eye_features = [left_eye_features left_eyebrow_features];
% [left_eye_patch, ~, coords] = Get_SR_Patch(sr_image, input, left_eye_features, landmarks_sr, dataset_landmarks, "left_eye_center", 10, 3);
% left_eye_patch = imhistmatch(left_eye_patch, patched_image(coords(3):coords(4),coords(1):coords(2)), 'method', 'polynomial');
% patched_image(coords(3):coords(4),coords(1):coords(2)) = left_eye_patch;
% 
% right_eye_features = [right_eye_features right_eyebrow_features];
% [right_eye_patch, ~, coords] = Get_SR_Patch(sr_image, input, right_eye_features, landmarks_sr, dataset_landmarks, "right_eye_center", 10, 3);
% right_eye_patch = imhistmatch(right_eye_patch, patched_image(coords(3):coords(4),coords(1):coords(2)), 'method', 'polynomial');
% patched_image(coords(3):coords(4),coords(1):coords(2)) = right_eye_patch;

% subplot(2,2,1)
% imshow(uint8(hr));
% subplot(2,2,2);
% imshow(sr_image);
% subplot(2,2,3);
% imshow(bicubic_image);
% subplot(2,2,4);
% imshow(patched_image);
% waitforbuttonpress
% size(hr)

% Mouth patch psnr's/ssim's
% subplot(2,2,1);
% imshow(uint8(hr_mouth));
% subplot(2,2,2);
% imshow(bicubic_mouth);
% subplot(2,2,3);
% imshow(sr_mouth);
% subplot(2,2,4);
% imshow(mouth_patch);
% waitforbuttonpress
% fprintf('%s Mouth - Patched PSNR %2.5f, SSIM %2.5f\n', '', csnr(mouth_patch, uint8(hr_mouth),0,0), ssim(mouth_patch, uint8(hr_mouth), 'Exponents', [0 0 1]) );
% fprintf('%s Mouth - Bicubic PSNR %2.5f, SSIM %2.5f\n', '', csnr(bicubic_mouth, uint8(hr_mouth),0,0), ssim(bicubic_mouth, uint8(hr_mouth), 'Exponents', [0 0 1]) );
% fprintf('%s Mouth - SR PSNR %2.5f, SSIM %2.5f\n', '', csnr(sr_mouth, uint8(hr_mouth),0,0), ssim(sr_mouth, uint8(hr_mouth), 'Exponents', [0 0 1]) );

% Set the input basename of the file
[~,name,ext] = fileparts(input);
input = strcat(name,ext);
imwrite(uint8(patched_image), ['Result/LBF_', input]);
end