%   Simulation for the solution of the total least squares problem as in
%   [1]. The original solution from [1] and the sketched solution [2] was
%   compared.

%   Reference:
%   [1] G.H. Golub, C.F. van Loan, "An analysis of the total least squares
%   problem", SIAM J. Numer. Anal 17 (1980), 883-893
%   [2] Yuji Nakatsukasa, Taejun Park, "A Fast Algorithm for Computing an
%   Approximate Null Space", arXiv, 2022

% dimension size of A (mxn) and B (mxk) - varying m
m = 2.^(14:18); m1 = size(m,2);
n = 1e3;
k = 10;

% no. of iterations (for avg. time)
iter = 1;

% right singular vectors - VA, sing. vals. - SA
VA = orth(randn(n,n)); SA = logspace(0,-3,n);

% original TLS solution 1) time taken, 2) TLS error (Frobenius)
times1 = zeros(m1,iter);
TLSerr1 = zeros(m1,iter);

% sketched TLS solution 1) time taken, 2) TLS error (Frobenius)
times2 = zeros(m1,iter);
TLSerr2 = zeros(m1,iter);

% Comparisons: 1) relative residual (Frobenius), 2) relative error (2-norm),
% 3) sine of the angle between the original sol. and the sketched sol.
relres = zeros(m1,iter);
relerr = zeros(m1,iter);
sinang = zeros(m1,iter);

for i = 1:m1
    for j = 1:iter
        
        % mxn matrix A with sing. val. = (SA) and sing. vec. Haar distributed
        A = orth(randn(m(i),n))*diag(SA)*VA';
        
        % B in the direction of A with some error of size 1e-8
        B = A*randn(n,k)/sqrt(n)+randn(m(i),k)*1e-8/sqrt(m(i));
        
        % augmented matrix Z
        Z = [A,B];
        
        % original TLS solution
        tic
        [U,S,V] = svd(Z,0);
        X = -V(1:n,n+1:end)/V(n+1:end,n+1:end);
        times1(i,j) = toc;
        
        % sketched TLS solution
        tic
        SZ = SRFT(Z,2);
        [US,SS,VS] = svd(SZ,0);
        SX = -VS(1:n,n+1:end)/VS(n+1:end,n+1:end);
        times2(i,j) = toc;
        
        % TLS error
        err1 = -1* U(:,n+1:end) * S(n+1:end,n+1:end) * V(:,n+1:end)';
        TLSerr1(i,j) = norm(err1,'f');
        
        % sketched TLS error
        RHS = B-A*SX;
        LHS = [SX;-eye(k)];
        [q,r] = qr(LHS,0);
        err2 = RHS/r*q';
        TLSerr2(i,j) = norm(err2,'f');
        
        % 1) rel. res., 2) rel. err., 3) sine of angle
        relres(i,j) = TLSerr2(i,j)/TLSerr1(i,j);
        relerr(i,j) = norm(X-SX)/norm(X);
        sinang(i,j) = sin(subspace(X,SX));

    end
end
