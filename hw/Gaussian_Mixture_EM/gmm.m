¡¢
function varargout = gmm(X, model, K)
    [N, D] = size(X);
    threshold = 1e-2;
    % initial values
    pMiu = model.Miu;
    pSigma = model.Sigma;
    pPi = model.Pi;
 
    Lprev = -inf;
    while true
        Px = calc_prob();
 
        % new value for pGamma
        pGamma = Px .* repmat(pPi, N, 1);
        pGamma = pGamma ./ repmat(sum(pGamma, 2), 1, K);
 
        % new value for parameters of each Component
        Nk = sum(pGamma, 1);
        pMiu = diag(1./Nk) * pGamma' * X;
        pPi = Nk/N;
        for kk = 1:K
            Xshift = X-repmat(pMiu(kk, :), N, 1);
            pSigma(:, kk) = (Xshift' * ...
                (diag(pGamma(:, kk)) * Xshift)) / Nk(kk);
        end
 
        % check for convergence
        L = sum(log(Px*pPi'));
        if L-Lprev < threshold
            break;
        end
        Lprev = L;
    end
 
    if nargout == 1
        varargout = {Px};
    else
        model = [];
        model.Miu = pMiu;
        model.Sigma = pSigma;
        model.Pi = pPi;
        varargout = {Px, model};
    end
 
    function Px = calc_prob()
        Px = zeros(N, K);
        for k = 1:K
            Xshift = X-repmat(pMiu(k, :), N, 1);
            inv_pSigma = inv(pSigma(:, k));
            tmp = sum((Xshift*inv_pSigma) .* Xshift, 2);
            coef = (2*pi)^(-1/2) * sqrt(det(inv_pSigma));
            Px(:, k) = coef * exp(-0.5*tmp);
        end
    end
end
