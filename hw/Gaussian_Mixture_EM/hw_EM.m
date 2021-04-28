warning off
load('homework.mat');

X = samples(1:1000);
X = X';
M = 5;
%≥ı ºªØ
[N, D] = size(X);
if isscalar(M)
    K = M;
    rndp = randperm(N);
    centroids = X(rndp(1:K),:);
else
    K = size(M, 1);
        centroids = M;
    end

pMiu = centroids;
pPi = zeros(1, K);
pSigma = zeros(1, K);
 
distmat = repmat(sum(X.*X, 2), 1, K) + ...
            repmat(sum(pMiu.*pMiu, 2)', N, 1) - ...
            2*X*pMiu';
[~, labels] = min(distmat, [], 2);
 
for k=1:K
    Xk = X(labels == k, :);
    pPi(k) = size(Xk, 1)/N;
    pSigma(1, k) = cov(Xk);
end

model = [];
model.Miu = pMiu;
model.Sigma = pSigma;
model.Pi = pPi;

[PX, model] = gmm(X, model, M);
fprintf('finish %d training batch.\n',1);
for i=1:99
    X = samples((1000*i+1):(1000*i+1000));
    X = X';
    [PX, model] = gmm(X, model, M);
    fprintf('finish %d training batch.\n',i+1);
end
