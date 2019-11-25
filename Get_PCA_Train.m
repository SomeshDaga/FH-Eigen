function [El, Eh, mY, mX, Y, X, Vl, Dh] = Get_PCA_Train(par, train_idxs, dataset)
%calculate PCA of training faces 

img_num = length(train_idxs);

for i = 1 : img_num
    imHR  =  imread(dataset(train_idxs(i)).file);
    [imHR, imLR] = get_LR( imHR, par );
    Y(:, i) = imLR(:); 
    X(:, i) = imHR(:);
end
mY = mean(Y, 2);
mX = mean(X, 2);
Y = Y - repmat(mY, [1, img_num]);
X = X - repmat(mX, [1, img_num]);

%calculate PCA coefficent
[El, Eh, Dl, Dh, Vl] = cal_PCA( double(Y), double(X), par.k );

end