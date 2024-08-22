clear;
clc;


addpath(genpath('./'));

datadir='/Datasets/';
Dataname = cell(2, 1);         %

Dataname{1} = 'NGs';


numdata = length(Dataname);
numname = {'_Per0.5'};


for idata = 1                                 
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    for dataIndex = 1:1:1 

        datafile = [cell2mat(Dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1, ind] = findindex(data, index);
        
        time1 = toc;
        maxAcc = 0;
        TempAnchor = [2*k];
        
        ACC = zeros(length(TempAnchor));
        NMI = zeros(length(TempAnchor));
        Purity = zeros(length(TempAnchor));
        idx = 1;
         beta = 2^-4;
         gamma = 2^2;

        for LambdaIndex2 = 1 : length(TempAnchor)
         
            numanchor = TempAnchor(LambdaIndex2);
            disp([char(Dataname(idata)), char(numname(dataIndex)),  '-numanchor=', num2str(numanchor)]);
            tic;
            [F,B,W,Z,S,iter,obj,alpha,ts,X_complete] = PBDAL(X1,gt,3*k,numanchor,beta,gamma); 
            F = F ./ (repmat(sqrt(sum(F .^ 2, 2)), 1, k)+eps);

            time2 = toc;
            stream = RandStream.getGlobalStream;
            reset(stream);
            MAXiter = 1000; 
            REPlic = 20; 
            tic;
            for rep = 1 : 20
                pY = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                res(rep, : ) = Clustering8Measure(gt, pY);
            end
            time3 = toc;
            runtime(idx) = time1 + time2 + time3/20;
            disp(['runtime:', num2str(runtime(idx))])
            idx = idx + 1;
            tempResBest(dataIndex, : ) = mean(res);
            tempResStd(dataIndex, : ) = std(res);
            for tempIndex = 1 : 8
                if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                    if tempIndex == 1
                        newZ = Z;
                        newF = F;
                    end
                    ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                    ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                end
            end
        end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
    end
end

