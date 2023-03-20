function K = gaussian_kernel(A, B, sigma)
    n = size(A, 1);
    p = size(B, 1);
    
    repeatMatrix = @(x, n, p) repmat(x, [n, p]);
    
    if(size(A, 2) == size(B, 2))
        rowAnorm = sum(A.^2, 2);
        AnormMatrix = repeatMatrix(rowAnorm, n, p);

        colBnorm = sum(B.^2, 2)';
        BnormMatrix = repeatMatrix(colBnorm, n, p);

        K = exp(-sigma*(AnormMatrix + BnormMatrix - 2*A*B'));
    else
        error('ERROR: non-conformable arguments');
    end
end
