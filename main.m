%implement "Hallucinating Face by Eigentransformation
%by Li Yanghao

%clear all;clc;
addpath('Utilities');
addpath('Utilities/urlreadpost');
dataset_info = load('landmarks_cafe.mat');

%parameters
par.nFactor = 5;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
par.alpha = 0.05;   %parameter in second condition
par.k = 50; % No of eigenfaces to use

% Perform a 80-20 split for the dataset
% and create 5 folds such that all data gets a chance to be included
% in both training and testing sets
dataset = cvpartition(length(dataset_info.data), 'KFold', 5); %length(dataset_info.data));

face_recognition_results = zeros(length(sum(dataset.TestSize)),2);
update_string_length = 0;
for fold=1:dataset.NumTestSets
    train_idxs = find(dataset.training(fold));
    test_idxs = find(dataset.test(fold));
    [El, Eh, mY, mX, Y, X, Vl, Dh] = Get_PCA_Train(par, train_idxs, dataset_info.data);
    for i=1:length(test_idxs)
        idx = i + sum(dataset.TestSize(1:fold-1));
%         for j=1:update_string_length
%             fprintf('\b');
%         end
        update_string = sprintf("Processing %d of %d\n", idx, sum(dataset.TestSize));
        update_string_length = strlength(update_string);
        fprintf(update_string);
        test_image = dataset_info.data(test_idxs(i)).file;
        [bicubic, sr, hr] = SR_by_PCA(par, test_image, El, Eh, mY, mX, X, Vl, Dh );

        % Extract landmarks from super-resolved image
        landmark_sr = get_landmarks(sr, false);
        landmark_bicubic = get_landmarks(bicubic, false);

        % Use extracted landmarks to create another super-resolved image
%         patched_image = SR_by_LBF( par, test_image, hr, sr, bicubic, landmark_sr, landmark_bicubic, dataset_info.data(train_idxs), mX);
        patched_image = SR_by_LBF( par, test_image, hr, sr, bicubic, landmark_sr, landmark_bicubic, dataset_info.data(train_idxs), mX);
        
        % Compare the face recognition thresholds
        sr_confidence = get_face_similarity(uint8(sr), uint8(hr));
        patched_confidence = get_face_similarity(uint8(patched_image), uint8(hr));
        face_recognition_results(idx, :) = [sr_confidence patched_confidence];
        fprintf("Recognition Confidence: SR: %f%%, Patched %f%%\n", sr_confidence, patched_confidence);
    end
end

% Compute statistics about the face recognition results
recognition_improvements = face_recognition_results(:,2) - face_recognition_results(:,1);
improved_idxs = recognition_improvements > 0;
worsened_idxs = ~improved_idxs;
mean_improvement = mean(recognition_improvements);
num_improved = sum(improved_idxs);
num_worsened = sum(worsened_idxs);
mean_of_improved = mean(recognition_improvements(improved_idxs));
mean_of_worsened = mean(recognition_improvements(worsened_idxs));
std_of_improved = std(recognition_improvements(improved_idxs));
std_of_worsened = std(recognition_improvements(worsened_idxs));
fprintf("\n\n--- Face Recognition Statistics ---\n\n");
fprintf("Net Improvement: %f%%\n", mean(recognition_improvements));
fprintf("Improved: %d/%d, Mean Improvement: %f%%, Std Dev Improvement: %f%%\n", num_improved, length(recognition_improvements), mean_of_improved, std_of_improved);
fprintf("Worsened: %d/%d, Mean Worsened: %f%%, Std Dev Worsened: %f%%\n", num_worsened, length(recognition_improvements), mean_of_worsened, std_of_worsened);