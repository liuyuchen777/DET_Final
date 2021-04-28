warning off
load('homework.mat');

X = samples(1:10000);
X = X';
M = 5;
[N D] = size(X);
% initialize
Miu = rand([M 1])*(max(X)-min(X))+min(X);
Sigma = diag(rand([M 1]));
Alpha(1:M,1) = 1/M;

threshold = 1e-5;
Lprev = -inf;
i=1;
while true
    fprintf('finish NO.%d training.\n',i);
    i = i+1;
    % 计算每一个估计值属于某一个高斯模型的概率
    Px = calc_prob(Miu, X, Sigma, N, M);
    
    % 更新各个参数
    Alpha = (sum(Px,1)/N)';
    Miu = Px'*X./sum(Px, 1)';
    for m = 1:M
        Xshift = X-repmat(Miu(m), N, 1);
        Sigma(m) = sum(Xshift * Xshift' * Px, 1) / sum(Px, 1);    
    end
    
    % 检查是否收敛
    L = sum(log(Px*Alpha));
    if L-Lprev < threshold
        break;
    end
    Lprev = L;
end

function Px = calc_prob(Miu, X, Sigma, N, M)
    Px = zeros(N, M);
    for m = 1:M
        Xshift = X-repmat(Miu(m), N, 1);
        inv_Sigma = 1/Sigma(m, m);
        tmp = sum((Xshift*inv_Sigma) .* Xshift, 2);
        coef = (2*pi)^(-1/2) * sqrt(inv_Sigma);
        Px(:, m) = coef * exp(-0.5*tmp);
    end
end


