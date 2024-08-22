function [X,tnn,trank] = proxy_scahtten(Y,rho,p)
[n1,n2,n3] = size(Y);
X = zeros(n1,n2,n3);
Y = fft(Y,[],3);
tnn = 0;
trank = 0;
epsilon=1e-16;
C=1;
%% first frontal slice
[U,S,V] = svd(Y(:,:,1),'econ');
S = diag(S)';
r = length(find(S>rho));
if r>=1
    %     S = S(1:r)-rho;
    w_p=C./(S(1:r)+epsilon);
    S = solve_Lp_w(S(1:r),w_p,p);
    X(:,:,1) = U(:,1:r)*diag(S)*V(:,1:r)';
    tnn = tnn+sum(S);
    trank = max(trank,r);
end
%% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S)';
    r = length(find(S>rho));
    if r>=1
        %         S = S(1:r)-rho;
        w_p=C./(S(1:r)+epsilon);
        S = solve_Lp_w(S(1:r),w_p,p);
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S)*2;
        trank = max(trank,r);
    end
    X(:,:,n3+2-i) = conj(X(:,:,i));
end
%% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    [U,S,V] = svd(Y(:,:,i),'econ');
    S = diag(S)';
    r = length(find(S>rho));
    if r>=1
        %         S = S(1:r)-rho;
        w_p=C./(S(1:r)+epsilon);
        S = solve_Lp_w(S(1:r),w_p,p);
        X(:,:,i) = U(:,1:r)*diag(S)*V(:,1:r)';
        tnn = tnn+sum(S);
        trank = max(trank,r);
    end
end
%%
tnn = tnn/n3;
X = ifft(X,[],3);
end

