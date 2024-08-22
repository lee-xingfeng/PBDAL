function [res,n1,n2] = tnn_scahtten_p_w(X,Y,rho,p)

% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% Y     -    n1*n2*n3 tensor
%min_X rho*||X||_{w,S_p}^p+0.5*||X-Y||_F^2 %modified by Xingfeng Li 
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X

[n1,n2,n3] = size(Y);
res = zeros(n1,n2,n3);
halfn3 = round(n3/2);
Y = fft(Y,[],3);
X = fft(X,[],3);
epsilon=1e-16;
C=1;

%% first frontal slice
[U_f,S_f,V_f] = svd(Y(:,:,1),'econ');%
S_f = diag(S_f);                     %
[~,S_x,~] = svd(X(:,:,1),'econ');
S_x = diag(S_x);
w_p=C./(S_x+epsilon);
D_f = solve_Lp_w(S_f, rho.*w_p, p);
res(:,:,1) = U_f*diag(D_f)*V_f';

for i = 2 : halfn3
    [U_f,S_f,V_f] = svd(Y(:,:,i),'econ');
    S_f = diag(S_f);
    [~,S_x,~] = svd(X(:,:,i),'econ');%
    S_x = diag(S_x);
    w_p=C./(S_x+epsilon);
    D_f = solve_Lp_w(S_f, rho.*w_p, p);
    res(:,:,i) = U_f*diag(D_f)*V_f';
    res(:,:,n3+2-i) = conj(res(:,:,i));
end
%%
if mod(n3,2) == 0
    i = halfn3+1;
    [U_f,S_f,V_f] = svd(Y(:,:,i),'econ');
    S_f = diag(S_f);
    [~,S_x,~] = svd(X(:,:,i),'econ');
    S_x = diag(S_x);
    w_p=C./(S_x+epsilon);
    D_f = solve_Lp_w(S_f, rho.*w_p, p);
    res(:,:,i) = U_f*diag(D_f)*V_f';
end
res = ifft(res,[],3);
