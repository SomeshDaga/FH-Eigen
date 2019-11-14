%implement "Hallucinating Face by Eigentransformation
%by Li Yanghao

%clear all;clc;
addpath('Utilities');

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

SR_by_PCA( par, El, Eh, mY, mX, X, Vl, Dh );
