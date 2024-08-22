function [X,tnn,trank] = prox_n_w_itnn(shiftTen,Y, tao,p)
    % process improved tensor nuclear norm
    [n1,n2,n3] = size(Y);
    n12 = min(n1,n2);
  
    % get core matrix
    lambda1 = 1/sqrt(max(n12,n3));
    [U, S, V] = n_tsvd(Y,'econ');
    reshapeS = zeros(n12, n3);
    for i = 1:n12
        reshapeS(i,:) = S(i,i,:);
    end
  
    [A,D,B] = svd(reshapeS, 'econ');
    VT=B';
    D = diag(D);
    ind = find(D > lambda1);
    D = diag(D(ind) - lambda1);
    L = A(:,ind) * D * VT(ind,:);
    T = zeros(n12,n12,n3);
    for i = 1:n12
        T(i,i,:) = L(i,:);
    end
    T = tprod(tprod(U,T), tran(V));
    temp_T = T;

    [X,tnn,trank] = tnn_scahtten_p_w(temp_T,T, tao,p);
end

