function [K] = poly_kernel(A, B, d)
% Translates R function poly_kernel to MATLAB

% Convert inputs to matrices
A = double(A);
B = double(B);

% Get dimensions
n = size(A, 1);
p = size(B, 2);

% Compute kernel matrix
if size(A, 2) == size(B, 1)
K = (A * B').^d;
else
error('ERROR: non-conformable arguments');
end

end