function gnhClassifier = lsgnhsvm(trainx, trainy, lam1, lam2, kernel, varargin)
% compute the kernel
if strcmp(kernel, 'linear')
x_pos = trainx(trainy == 1, :);
x_neg = trainx(trainy == -1, :);
elseif strcmp(kernel, 'gaussian')
xk_pos = trainx(trainy == 1, :);
xk_neg = trainx(trainy == -1, :);
xk = [xk_pos; xk_neg];
x_pos = gaussian_kernel(xk_pos, xk', varargin{:});
x_neg = gaussian_kernel(xk_neg, xk', varargin{:});
elseif strcmp(kernel, 'poly')
xk_pos = trainx(trainy == 1, :);
xk_neg = trainx(trainy == -1, :);
xk = [xk_pos; xk_neg];
x_pos = poly_kernel(xk_pos, xk', varargin{:});
x_neg = poly_kernel(xk_neg, xk', varargin{:});
end

n_pos = size(x_pos, 1);% 正类的样本个数
n_neg = size(x_neg, 1);%负类的样本个数
n = n_pos + n_neg;
p = size(x_pos, 2);% 特征的个数

e_pos = ones(n_pos, 1);% 正类的单位向量
e_neg = ones(n_neg, 1);% 负类的单位向量
e = ones(n, 1);

x_tilde_pos = [x_pos e_pos];
x_tilde_neg = [x_neg e_neg];
x_tilde = [x_tilde_pos; -x_tilde_neg];

% compute matrix Q
E = eye(p + 1);
mat1 = pinv(E + lam1*x_tilde_pos'*x_tilde_pos);
mat2 = pinv(E + lam1*x_tilde_neg'*x_tilde_neg);
Q = x_tilde*(mat1 + mat2)*x_tilde';

% compute u
u = pinv(Q + diag(n)/lam2)*e;

% compute w+, b+ and w-, b-
w_tilde_pos = mat1'*x_tilde'*u;
w_tilde_neg = mat2'*x_tilde'*u;
w_pos = w_tilde_pos(1:p);
b_pos = w_tilde_pos(p+1);
w_neg = w_tilde_neg(1:p);
b_neg = w_tilde_neg(p+1);

gnhClassifier.wpos = w_pos;
gnhClassifier.bpos = b_pos;
gnhClassifier.wneg = w_neg;
gnhClassifier.bneg = b_neg;
gnhClassifier.u = u;
end

