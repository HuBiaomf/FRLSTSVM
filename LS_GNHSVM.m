clear;
rng(3);
warning off
dataset_name = ["diabetes" "Australian"   "CMC"       "vote_d"    "ionosphere" ...% 5
                 "iris"     "glass"    "hepatitis"   "Heart-c"   "Heart-Statlog" ...% 10
                 "wine"    "soybean-small"...% 15
                 "cleveland"     "tae"       "mushdata"   ];
% soy iris tae hep wine glass heart-s  cleve heart-c iono vote aus dia cmc mush
%mypar = parpool;
tic
for d = 15:15
    path = strcat("Data Sets\",dataset_name(d),".csv");
    MainData = readmatrix(path); 
    X = MainData(:,1:(end-1));
    Y = MainData(:,end);
    iter = 10;bia =  4;
    acc_gather = zeros(iter,iter);
    std_gather = acc_gather;
    for h = 1:iter
        c1 = 2.^(h-bia);
        for t = 1:iter
            c2 = 2.^(t-bia);
                KfoldNumber = 10;
                indices = crossvalind('Kfold',Y,KfoldNumber);
                acc = zeros(KfoldNumber,1);

                for i = 1:KfoldNumber
                    test = (indices == i);
                    train = ~test;
                    Xtrain = X(train==1,:);        Ytrain = Y(train==1);
                    Xtest = X(test==1,:);          Ytest = Y(test==1);
                    m3 = size(Xtest,1);
                    gnhClassifier = lsgnhsvm(Xtrain, Ytrain, c1, c2, "linear", 0);
                    w1 = gnhClassifier.wpos;
                    w2 = gnhClassifier.wneg;
                    b1 = gnhClassifier.bpos;
                    b2 = gnhClassifier.bneg;
                    result = ones(m3,1);
                    for ii=1:m3
                        mu1 = abs((Xtest(ii,:)*w1 + b1)/norm(w1,2));
                        mu2 = abs((Xtest(ii,:)*w2+b2)/norm(w2,2));
                    
                        if mu1 < mu2
                            result(ii) = 1;
                        else
                            result(ii) = -1;
                        end
                    end
                    accsum = 0;
                    for ii = 1:m3
                        if(result(ii)==Ytest(ii))
                            accsum = accsum + 1;
                        end
                    end
                    acc(i) = accsum/m3;
                end
                acc_gather(h,t) = mean(acc);
                std_gather(h,t) = std(acc);
                %fprintf("h:%d,t:%d\n",h,t);
        end
    end
    max_acc = 0;
    for h = 1:iter
        for t = 1:iter
            if(acc_gather(h,t)>max_acc)
                max_acc = acc_gather(h,t);
                max_std = std_gather(h,t);
                max_c1 = h;max_c2 = t;
                fprintf("h = %d,t = %d,\n",h,t);
            end
        end
    end
    fprintf("h,t = %d,%d,数据集：%s,acc = %d\n,std = %d\n",max_c1,max_c2,dataset_name(d),max_acc,max_std);
end
toc
