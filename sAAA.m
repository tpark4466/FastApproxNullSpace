function [r, pol, res, zer, zj, fj, wj, errvec, wt] = sAAA(F, varargin)
%   Sketched AAA with reused sketch. (Lawson is turned off [3])
%   The code for sketched AAA with reused sketch is written on top of the
%   original AAA code from Chebfun. The differences are in lines 75 - 90.
%   The original code aaa.m is freely available at http://www.chebfun.org/.

%   References for sketched AAA:
%   [1] Yuji Nakatsukasa, Taejun Park, "A Fast Algorithm for Computing an
%   Approximate Null Space", arXiv, 2022

%   References for original AAA:
%   [2] Yuji Nakatsukasa, Olivier Sete, Lloyd N. Trefethen, "The AAA algorithm
%   for rational approximation", SIAM J. Sci. Comp. 40 (2018), A1494-A1522.
%
%   [3] Yuji Nakasukasa and Lloyd N. Trefethen, An algorithm for real and
%   complex rational minimax approximation, arXiv, 2019.

% Parse inputs:
[F, Z, M, dom, tol, mmax, cleanup_flag, cleanup_tol, needZ, mmax_flag, nlawson] ...
    = parseInputs(F, varargin{:});

if ( needZ )
    % Z was not provided.  Try to resolve F on its domain.
    [r, pol, res, zer, zj, fj, wj, errvec] = ...
        aaa_autoZ(F, dom, tol, mmax, cleanup_flag, cleanup_tol, mmax_flag, nlawson);
    return
end

% Remove any infinite or NaN function values (avoid SVD failures):
toKeep = ~isinf(F);
F = F(toKeep); Z = Z(toKeep);
toKeep = ~isnan(F);
F = F(toKeep); Z = Z(toKeep);

% Remove repeated elements of Z and corresponding elements of F:
[Z, uni] = unique(Z,'stable'); F = F(uni);

M = length(Z);

% Relative tolerance:
reltol = tol * norm(F, inf);

% Left scaling matrix:
SF = spdiags(F, 0, M, M);

% Initialization for AAA iteration:
J = 1:M;
zj = [];
fj = [];
C = [];
errvec = [];
R = mean(F);
SA = [];
sketch_size = 200;
DD = sign(randn(M, 1));
IX = randsample(M,sketch_size);
JZ = [];
A = [];

% AAA iteration:
for m = 1:mmax
    % Select next support point where error is largest:
    [~, jj] = max(abs(F - R));          % Select next support point.       %nec
    zj = [zj; Z(jj)];                   % Update support points.           %nec
    fj = [fj; F(jj)];                   % Update data values.              %nec
    J(J == jj) = [];                    % Update index vector.             %nec
    C = [C 1./(Z - Z(jj))];             % Next column of Cauchy matrix.
    JZ = [JZ,jj];
    
    % Compute weights:
    Sf = diag(fj);                      % Right scaling matrix.
    
    if m>1
        prevA = A(jj,:);                % previous row deletion
        A(jj,:) = 0;
    end
    
    Am = SF*C(:,end)-C(:,end)*fj(end);
    Am(JZ,:) = 0;
    A = [A,Am];                         % Modified - Loewner matrix update
    if m== 1
        SA = fft(DD.*A);
        SA = SA(IX,:);                  % initial sketch using SRFT
    else    
        temp = DD(jj)*fft((1:M==jj)');  % row downdate (reusing sketch)
        temp2 = fft(DD.*A(:,end));      % column update (reusing sketch)
        SA = [SA-temp(IX,:)*prevA temp2(IX,:)]; % row downdate + column update
    end
    
    [~,~,V] = svd(SA,0); 
    
    wj = V(:,m);                        % weight vector = min sing vector
    
    % Rational approximant on Z:
    N = C*(wj.*fj);                     % Numerator
    D = C*wj;                           % Denominator
    R = F;
    R(J) = N(J)./D(J);
    
    % Error in the sample points:
    maxerr = norm(F - R, inf);
    errvec = [errvec; maxerr];

    % Check if converged:
    if ( maxerr <= reltol )
        break
    end
end
maxerrAAA = maxerr;                     % error at end of AAA 

% When M == 2, one weight is zero and r is constant.
% To obtain a good approximation, interpolate in both sample points.
if ( M == 2 )
    zj = Z;
    fj = F;
    wj = [1; -1];       % Only pole at infinity.
    wj = wj/norm(wj);   % Impose norm(w) = 1 for consistency.
    errvec(2) = 0;
    maxerrAAA = 0;
end

% We now enter Lawson iteration: barycentric IRLS = iteratively reweighted
% least-squares if 'lawson' is specified with NLAWSON > 0 or 'mmax' is
% specified and 'lawson' is not.  In the latter case the number of steps
% is chosen adaptively.  Note that the Lawson iteration is unlikely to be
% successful when the errors are close to machine precision.

wj0 = wj; fj0 = fj;     % Save parameters in case Lawson fails
wt = NaN(M,1); wt_new = ones(M,1);
nlawson = 0;
if ( nlawson > 0 )      % Lawson iteration

    maxerrold = maxerrAAA;
    maxerr = maxerrold;
    nj = length(zj);
    A = [];
    for j = 1:nj                              % Cauchy/Loewner matrix
        A = [A 1./(Z-zj(j)) F./(Z-zj(j))];
    end
    for j = 1:nj
        [i,~] = find(Z==zj(j));               % support pt rows are special
        A(i,:) = 0;
        A(i,2*j-1) = 1;
        A(i,2*j) = F(i);
    end
    sketch_size = 2*size(A,2);
    DD = sign(randn(size(A,1),1));
    %SA1 = fft(DD.*A);
    SA1 = DD.*A;
    IX = randsample(size(A,1),sketch_size);
    stepno = 0;
    while ( (nlawson < inf) & (stepno < nlawson) ) |...
          ( (nlawson == inf) & (stepno < 20) ) |...
          ( (nlawson == inf) & (maxerr/maxerrold < .999) & (stepno < 1000) ) 
        stepno = stepno + 1;
        wt = wt_new;
        W = spdiags(sqrt(wt),0,M,M);
        %SA2 = W*SA1;
        SA2 = fft(W*SA1);
        [~,~,V] = svd(SA2(IX,:),0);
%         if size(W,1) > 2*size(A,2)
%             WSA = SRFT(W*A,2);
%             [~, ~, V] = svd(WSA, 0);         % Reduced SVD.
%         else
%             [~, ~, V] = svd(W*A, 0);
%         end
        c = V(:,end);
        denom = zeros(M,1); num = zeros(M,1);
        for j = 1:nj
            denom = denom + c(2*j)./(Z-zj(j));
            num = num - c(2*j-1)./(Z-zj(j));
        end
        R = num./denom;
        for j = 1:nj
            [i,~] = find(Z==zj(j));           % support pt rows are special
            R(i) = -c(2*j-1)/c(2*j);
        end
        err = F - R; abserr = abs(err);
        wt_new = wt.*abserr; wt_new = wt_new/norm(wt_new,inf);
        maxerrold = maxerr;
        maxerr = max(abserr);
    end
    wj = c(2:2:end);
    fj = -c(1:2:end)./wj;
    % If Lawson has not reduced the error, return to pre-Lawson values.
    if (maxerr > maxerrAAA) & (nlawson == Inf)
        wj = wj0; fj = fj0; 
    end
end

% Remove support points with zero weight:
I = find(wj == 0);
zj(I) = [];
wj(I) = [];
fj(I) = [];

% Construct function handle:
r = @(zz) reval(zz, zj, fj, wj);

% Compute poles, residues and zeros:
[pol, res, zer] = prz(r, zj, fj, wj);

if ( cleanup_flag & nlawson == 0)       % Remove Froissart doublets
    [r, pol, res, zer, zj, fj, wj] = ...
        cleanup(r, pol, res, zer, zj, fj, wj, Z, F, cleanup_tol);
end

end % of AAA()


%% parse Inputs:

function [F, Z, M, dom, tol, mmax, cleanup_flag, cleanup_tol, ...
    needZ, mmax_flag, nlawson] = parseInputs(F, varargin)
% Input parsing for AAA.

% Check if F is empty:
if ( isempty(F) )
    error('CHEBFUN:aaa:emptyF', 'No function given.')
elseif ( isa(F, 'chebfun') )
    if ( size(F, 2) ~= 1 )
        error('CHEBFUN:aaa:nColF', 'Input chebfun must have one column.')
    end
end

% Sample points:
if ( ~isempty(varargin) && isfloat(varargin{1}) )
    % Z is given.
    Z = varargin{1};
    if ( isempty(Z) )
        error('CHEBFUN:aaa:emptyZ', ...
            'If sample set is provided, it must be nonempty.')
    end
    varargin(1) = [];
end

% Set defaults for other parameters:
tol = 1e-13;         % Relative tolerance.
mmax = 400;          % Maximum number of terms.
cleanup_tol = 1e-13; % Cleanup tolerance.
nlawson = Inf;       % number of Lawson steps (Inf means adaptive)
% Domain:
if ( isa(F, 'chebfun') )
    dom = F.domain([1, end]);
else
    dom = [-1, 1];
end
cleanup_flag = 1;   % Cleanup on.
mmax_flag = 0;      % Checks if mmax manually specified.
cleanup_set = 0;    % Checks if cleanup_tol manually specified.
% Check if parameters have been provided:
while ( ~isempty(varargin) )
    if ( strncmpi(varargin{1}, 'tol', 3) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 1]) )
            tol = varargin{2};
            if ~cleanup_set & tol > 0 % If not manually set, set cleanup_tol to tol.
              cleanup_tol = tol;
            end
        end
        varargin([1, 2]) = [];
        
    elseif ( strncmpi(varargin{1}, 'degree', 6) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 1]) )
            if ( mmax_flag == 1 ) && ( mmax ~= varargin{2}+1 )
                error('CHEBFUN:aaa:degmmaxmismatch', ' mmax must equal degree+1.')
            end            
            mmax = varargin{2}+1;
            mmax_flag = 1;
        end
        varargin([1, 2]) = [];
        
    elseif ( strncmpi(varargin{1}, 'mmax', 4) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 1]) )            
            if ( mmax_flag == 1 ) && ( mmax ~= varargin{2})                
                error('CHEBFUN:aaa:degmmaxmismatch', ' mmax must equal degree+1.')
            end
            mmax = varargin{2};
            mmax_flag = 1;
        end
        varargin([1, 2]) = [];
        
    elseif ( strncmpi(varargin{1}, 'lawson', 6) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 1]) )
            nlawson = varargin{2};
        end
        varargin([1, 2]) = [];
        
    elseif ( strncmpi(varargin{1}, 'dom', 3) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 2]) )
            dom = varargin{2};
        end
        varargin([1, 2]) = [];
        if ( isa(F, 'chebfun') )
            if ( ~isequal(dom, F.domain([1, end])) )
                warning('CHEBFUN:aaa:dom', ...
                    ['Given domain does not match the domain of the chebfun.\n', ...
                    'Results may be inaccurate.'])
            end
        end
        
    elseif ( strncmpi(varargin{1}, 'cleanuptol', 10) )
        if ( isfloat(varargin{2}) && isequal(size(varargin{2}), [1, 1]) )
          cleanup_tol = varargin{2};
          cleanup_set = 1;
        end
        varargin([1, 2]) = [];

    elseif ( strncmpi(varargin{1}, 'cleanup', 7) )
        if ( strncmpi(varargin{2}, 'off', 3) || ( varargin{2} == 0 ) )
            cleanup_flag = 0;
        end
        varargin([1, 2]) = [];
        
    else
        error('CHEBFUN:aaa:UnknownArg', 'Argument unknown.')
    end
end


% Deal with Z and F:
if ( ~exist('Z', 'var') && isfloat(F) )
    % F is given as data values, pick same number of sample points:
    Z = linspace(dom(1), dom(2), length(F)).';
end

if ( exist('Z', 'var') )
    % Z is given:
    needZ = 0;
    
    % Work with column vector:
    Z = Z(:);
    M = length(Z);
    
    % Function values:
    if ( isa(F, 'function_handle') || isa(F, 'chebfun') )
        % Sample F on Z:
        F = F(Z);
    elseif ( isnumeric(F) )
        % Work with column vector and check that it has correct length.
        F = F(:);
        if ( length(F) ~= M )
            error('CHEBFUN:aaa:lengthFZ', ...
                'Inputs F and Z must have the same length.')
        end
    elseif ( ischar(F) )
        % F is given as a string input. Convert it to a function handle.
        F = str2op(vectorize(F));
        F = F(Z);
    else
        error('CHEBFUN:aaa:UnknownF', 'Input for F not recognized.')
    end
    
else
    % Z was not given.  Set flag that Z needs to be determined.
    % Also set Z and M since they are needed as output.
    needZ = 1;
    Z = [];
    M = length(Z);
end

if ~mmax_flag & (nlawson == Inf)
    nlawson = 0;               
end

end % End of PARSEINPUT().


%% Cleanup

function [r, pol, res, zer, z, f, w] = ...
    cleanup(r, pol, res, zer, z, f, w, Z, F, cleanup_tol) 
% Remove spurious pole-zero pairs.

% Find negligible residues:
ii = find(abs(res) < cleanup_tol * norm(F, inf));
ni = length(ii);
if ( ni == 0 )
    % Nothing to do.
    return
elseif ( ni == 1 )
    warning('CHEBFUN:aaa:Froissart','1 Froissart doublet');
else
    warning('CHEBFUN:aaa:Froissart',[int2str(ni) ' Froissart doublets']);
end

% For each spurious pole find and remove closest support point:
for j = 1:ni
    azp = abs(z-pol(ii(j)));
    jj = find(azp == min(azp),1);
    
    % Remove support point(s):
    z(jj) = [];
    f(jj) = [];
end

% Remove support points z from sample set:
for jj = 1:length(z)
    F(Z == z(jj)) = [];
    Z(Z == z(jj)) = [];
end
m = length(z);
M = length(Z);

% Build Loewner matrix:
SF = spdiags(F, 0, M, M);
Sf = diag(f);
C = 1./bsxfun(@minus, Z, z.');      % Cauchy matrix.
A = SF*C - C*Sf;                    % Loewner matrix.

% Solve least-squares problem to obtain weights:
if size(A,1) > 2*size(A,2)
    SA = SRFT(A,2);
    [~, ~, V] = svd(SA, 0);         % Reduced SVD.
else
    [~, ~, V] = svd(A, 0);
end
w = V(:,m);

% Build function handle and compute poles, residues and zeros:
r = @(zz) reval(zz, z, f, w);
[pol, res, zer] = prz(r, z, f, w);

end % End of CLEANUP().


%% Automated choice of sample set

function [r, pol, res, zer, zj, fj, wj, errvec] = ...
    aaa_autoZ(F, dom, tol, mmax, cleanup_flag, cleanup_tol, mmax_flag, nlawson)
%

% Flag if function has been resolved:
isResolved = 0;

% Main loop:
for n = 5:14
    % Sample points:
    % Next line enables us to do pretty well near poles
    Z = linspace(dom(1)+1.37e-8*diff(dom), dom(2)-3.08e-9*diff(dom), 1 + 2^n).';
    [r, pol, res, zer, zj, fj, wj, errvec] = aaa(F, Z, 'tol', tol, ...
        'mmax', mmax, 'cleanup', cleanup_flag, 'cleanuptol', cleanup_tol, 'lawson', nlawson);
    
    % Test if rational approximant is accurate:
    reltol = tol * norm(F(Z), inf);
    
    % On Z(n):
    err(1,1) = norm(F(Z) - r(Z), inf);
    
    Zrefined = linspace(dom(1)+1.37e-8*diff(dom), dom(2)-3.08e-9*diff(dom), ...
        round(1.5 * (1 + 2^(n+1)))).';
    err(2,1) = norm(F(Zrefined) - r(Zrefined), inf);
    
    if ( all(err < reltol) )
        % Final check that the function is resolved, inspired by sampleTest().
        % Pseudo random sample points in [-1, 1]:
        xeval = [-0.357998918959666; 0.036785641195074];
        % Scale to dom:
        xeval = (dom(2) - dom(1))/2 * xeval + (dom(2) + dom(1))/2;
        
        if ( norm(F(xeval) - r(xeval), inf) < reltol )
            isResolved = 1;
            break
        end
    end
end

if ( ( isResolved == 0 ) && ~mmax_flag )
    warning('CHEBFUN:aaa:notResolved', ...
        'Function not resolved using %d pts.', length(Z))
end

end % End of AAA_AUTOZ().

function op = str2op(op)
    % Convert string inputs to either numeric format or function_handles.
    sop = str2num(op);
    if ( ~isempty(sop) )
        op = sop;
    else
        depVar = symvar(op);
        if ( numel(depVar) ~= 1 )
            error('CHEBFUN:CHEBFUN:str2op:indepvars', ...
             'Incorrect number of independent variables in string input.');
        end
        op = eval(['@(' depVar{:} ')', op]);
    end
end % End of STR2OP().