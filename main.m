%implement "Hallucinating Face by Eigentransformation
%by Li Yanghao

%clear all;clc;
addpath('Utilities');
addpath('Utilities/urlreadpost');
dataset_landmarks = load('landmarks.mat');

%parameters
par.nFactor = 5;
par.psf =   fspecial('gauss', 7, 1.6);              % The simulated PSF
par.train_path = '../CAFE-FACS-aligned/';
par.train_type = '*.png';
par.test_path = '../Test';
par.test_type = '*.png';
par.alpha = 0.05;   %parameter in second condition
par.k = 10; % No of eigenfaces to use

[El, Eh, mY, mX, Y, X, Vl, Dh] = Get_PCA_Train( par );

img_path  = par.test_path;
img_type = par.test_type;
img_dir = dir( fullfile(img_path, img_type) );
img_num = length(img_dir);
for i=1:img_num
    test_image = fullfile(img_path, img_dir(i).name);
    [bicubic, sr] = SR_by_PCA( par, test_image, El, Eh, mY, mX, X, Vl, Dh );
    % Extract landmarks from super-resolved image
    landmark = get_landmarks(bicubic, false);
    
    % Use extracted landmarks to create another super-resolved image
    SR_by_LBF( par, sr, bicubic, landmark, dataset_landmarks.data, mX);
end
