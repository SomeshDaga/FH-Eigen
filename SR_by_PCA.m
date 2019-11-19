function [ bicubic, sr, imHR ] = SR_by_PCA( par, input, El, Eh, mY, mX, X, Vl, Dh )
%reconstruct face images

imHR  =  imread(input) ;
[imHR, imLR] = get_LR( imHR, par );
[im_h, im_w] = size( imHR );

c = Vl * (El' * (imLR(:) - mY));
x = real( X * c + mX );
result = reshape(x, [im_h, im_w]);

% Set the input basename of the file
[~,name,ext] = fileparts(input);
input = strcat(name,ext);

fprintf('%s FHE PSNR %2.5f, SSIM %2.5f\n', input, csnr(uint8(result), uint8(imHR),0,0), ssim(result, imHR, 'Exponents', [0 0 1]) );

wh = Eh'*(x-mX);
for j = 1:size(wh, 1)
    if abs(wh(j)) > par.alpha*sqrt(Dh(j,j))
        wh(j) = sign(wh(j))*par.alpha*sqrt(Dh(j,j));
    end
end
xh = real( Eh*wh + mX );
result = reshape(xh, [im_h, im_w]);

fprintf('%s FHE PSNR %2.5f, SSIM %2.5f\n', input, csnr(uint8(result), uint8(imHR),0,0), ssim(uint8(result), uint8(imHR), 'Exponents', [0 0 1]) );

imBicubic = imresize( imLR, par.nFactor, 'Bicubic');
fprintf('%s Bicubic PSNR %2.5f, SSIM %2.5f\n', input, csnr(uint8(imBicubic), uint8(imHR),0,0), ssim(uint8(imBicubic), uint8(imHR), 'Exponents', [0 0 1]) );


imwrite(uint8(reshape(imLR, [im_h/par.nFactor, im_w/par.nFactor])), ['Result/LR_', input]);
imwrite(uint8(result), ['Result/FHE_', input]);
imwrite(uint8(imBicubic), ['Result/Bicu_', input]);

% Seems like the face landmark localization is actually more accurate
% with the bicubic interpolated image than the super resolved ones
bicubic = uint8(imBicubic);
sr = uint8(result);
end

