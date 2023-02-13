function SA = SRFT(A,k)
    % SRFT sketch (left mult.) to mxn matrix A with sketch size kn
    % O(mnlog m) implementation
    
    n = size(A,1);
    s = k*size(A,2); % sketch dimension
    
    d = sign(randn(n, 1));
    if isreal(A)
        SA = dct(d .* A); SA(1, :) = SA(1, :) / sqrt(2);
    else
        SA = fft(d.*A); % if A complex
    end
    IX = randsample(n,s);
    SA = SA(IX,:)*sqrt(n/s);
end