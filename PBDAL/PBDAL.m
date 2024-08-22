function [UU,P,W,Z,S,iter,obj2,alpha,ts,X] = PBDAL(X,Y,d,numanchor,beta,gamma)
% m      : the number of anchor. the size of Z is m*n.
% X      : n*di
%% initialize
maxIter = 50 ; % the number of iterations
m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);

%% initialize 

for i = 1:numview
   P{i} = zeros(d,m); 
   di = size(X{i},1); 
   W{i} = zeros(di,d);
   Z{i}=zeros(m,numsample);% m  * n
   Z_old = Z;
   H{i}=zeros(m,numsample);% m  * n
   J{i}=zeros(m,numsample);% m  * n
end
   
alpha = ones(1,numview)/numview;
opt.disp = 0;

mu = 10e-5; max_mu = 10e10; pho_mu = 2;
flag = 1;
iter = 0;
%% construct F_ind 
    anchor  = m/numclass;
    similar_value=1/anchor;
    numdiag =  similar_value;
    numdim = anchor;
    a = toeplitz([numdiag,similar_value*ones(1,numdim-1)]); %
    b = repmat({a},numclass,1); 
    S_ind = blkdiag(b{:});
%%
while flag
    iter = iter + 1;
    
        %% optimize Z
    for v=1:numview
        temp_am{v} = (alpha(v)^2+mu/2)*ones(1,numsample); 
        temp_Z = 0;
        for p = 1:numview
            if p == v
                continue;
            else
                temp_Z = temp_Z + S_ind*Z_old{p};
            end
        end
        temp_PWX_ut{v} = (alpha(v)^2 * P{v}'*W{v}'*X{v}-0.5*beta*temp_Z+0.5*mu*(H{v} - J{v}/mu));
         for ii=1:numsample
            idx = 1:numanchor;
            ut = temp_PWX_ut{v}(idx,ii)./(temp_am{v}(ii));   
            Z{v}(idx,ii) = EProjSimplex_new(ut');
          end  
    end
    Z_old = Z;
        %% optimize H
         for i = 1:numview
             Z{i} = Z{i}';
             J{i} = J{i}';
         end
    Z_tensor = cat(3, Z{:,:});
    J_tensor = cat(3, J{:,:});
    Ten = Z_tensor+J_tensor/mu;
    Ten=shiftdim(Ten, 1);
    [tempH_tensor,~,~] = prox_n_itnn(Ten, gamma/mu);
    tempH_tensor = shiftdim(tempH_tensor, 2);
         for i = 1:numview
             Z{i} = Z{i}';
             J{i} = J{i}'; 
             H_tensor(:,:,i) = tempH_tensor(:,:,i)';
         end
    Z_tensor = cat(3, Z{:,:});
    J_tensor = cat(3, J{:,:});

    
    %% optimize W_i
    for v = 1:numview
        temp_W = alpha(v)^2 * X{v}*Z{v}'*P{v}';      
        [U,~,V] = svd(temp_W,'econ');
        W{v} = U*V';
    end

    %% optimize P{v}
    for v = 1:numview
    temp_P = W{v}' * X{v} * Z{v}';
    [Unew,~,Vnew] = svd(temp_P,'econ');
    P{v} = Unew*Vnew';
    end  
    
    %% obtain indicator from Z
    S=0;
    for v = 1:numview
        H{v} = H_tensor(:,:,v);
        J{v} = J_tensor(:,:,v);
       S = S+ alpha(v)^2 * Z{v};
    end
    [UU,~,V]=svd(S','econ');
    ts{iter} = UU(:,1:numclass);


     %% solve  Y_tensor and  penalty parameters
         
    J_tensor = J_tensor + mu*(Z_tensor - H_tensor);
    mu = min(mu*pho_mu, max_mu);
    
    term1 = 0;
    for v = 1:numview
        term1 = term1 + alpha(v)^2 * norm(W{v}'* X{v} - P{v} * Z{v},'fro')^2;
    end
    obj(iter) = term1;
    if (iter>1)
            obj2(iter) = abs((obj(iter-1)-obj(iter))/(obj(iter-1)));
    end
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(S','econ');
        UU = UU(:,1:numclass);
        flag = 0;
    end
end

         
         
    
