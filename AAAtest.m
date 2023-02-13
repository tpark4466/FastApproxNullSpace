% This file requires Chebfun to run.
% See http://www.chebfun.org/ for Chebfun information.

% number of points sampled
n = 1e6;

% no. of iterations (for avg. time)
iter = 1;

% function
f = @(z) sqrt(z.*(1-z)).*sqrt((z-1j).*(1+1j-z));

% x-values
x = rand(n,1)+1j*rand(n,1);

% data points
fdata = f(x);

t0 = zeros(iter,1); % AAA time
t1 = zeros(iter,1); % sketched AAA time (reusing sketch)


for i = 1:iter
    
    % original AAA - O(mn^3) complexity
    tic 
    [r,pol] = aaa(fdata,x,'tol',1e-12,'mmax',500); 
    t0(i) = toc;
    
    % sketched AAA (reusing the sketch) O(mnlog m +n^4) complexity
    tic
    [rxx,polxx] = sAAA(fdata,x,'tol',1e-12,'mmax',500);
    t1(i) = toc;
end

%% plotting
xymin = -1.2; % min. x, y values
xymax = 2.2;  % max. x, y values

% phase plot set up
g = @(z) log10(abs(r(z)-f(z)));
gx = @(z) log10(abs(rxx(z)-f(z)));
gr = linspace(xymin,xymax,round(sqrt(n)));
[mfRe, mfIm] = ndgrid(gr,gr);
mfg = g(complex(mfRe,mfIm));
mfgx = gx(complex(mfRe,mfIm));

subplot(2,2,1)
plot(x,'.')
hold on
plot(polxx,'.'), grid on, axis equal
title('$\textbf{Poles and Data points for AAA with reused sketch}$','interpreter','latex')
subplot(2,2,2)
plot_phase(rxx,[xymin,xymax,xymin,xymax]), axis on
title('$\textbf{Phase plot for AAA with reused sketch}$','interpreter','latex')
subplot(2,2,3)
contour(mfRe, mfIm, mfg), axis equal; colorbar
title('$\textbf{Contour plot of $\log_{10}(|f-r_1|)$}$','interpreter','latex')  % original AAA
subplot(2,2,4)
contour(mfRe, mfIm, mfgx), axis equal; colorbar
title('$\textbf{Contour plot of $\log_{10}(|f-r_2|)$}$','interpreter','latex') % reused sketch AAA