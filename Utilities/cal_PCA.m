function [El, Eh, Dl, Dh, Vl] = cal_PCA( Y, X, k )
%get PCA for large dimensions
    [V, Dl] = eig( Y'*Y );
    Dl = Dl(k:end,k:end);
    V = V(:,k:end);
    El = Y * V * Dl^(-0.5);
    Vl = V * Dl^(-0.5);   
    
    [V, Dh] = eig( X'*X );
    Dh = Dh(k:end,k:end);
    V = V(:,k:end);
    Eh = X * V * Dh^(-0.5);
end