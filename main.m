%implement "Hallucinating Face by Eigentransformation
%by Li Yanghao

%clear all;clc;
addpath('Utilities');
addpath('Utilities/urlreadpost');
dataset_info = load('landmarks.mat');

%parameters
par.nFactor = 5;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
par.alpha = 0.05;   %parameter in second condition
par.k = 50; % No of eigenfaces to use

% Perform a 80-20 split for the dataset
% and create 5 folds such that all data gets a chance to be included
% in both training and testing sets
dataset = cvpartition(length(dataset_info.data), 'KFold', 5);

for fold=1:dataset.NumTestSets
    train_idxs = find(dataset.training(fold));
    test_idxs = find(dataset.test(fold));
    [El, Eh, mY, mX, Y, X, Vl, Dh] = Get_PCA_Train(par, train_idxs, dataset_info.data);
    for i=1:length(test_idxs)
        test_image = dataset_info.data(test_idxs(i)).file;
        [bicubic, sr, hr] = SR_by_PCA(par, test_image, El, Eh, mY, mX, X, Vl, Dh );

        % Extract landmarks from super-resolved image
        landmark_bicubic = get_landmarks(bicubic, false);
        landmark_sr = get_landmarks(sr, false);

        % Use extracted landmarks to create another super-resolved image
        SR_by_LBF( par, test_image, hr, sr, bicubic, landmark_sr, landmark_bicubic, dataset_info.data(train_idxs), mX);
    end
end
