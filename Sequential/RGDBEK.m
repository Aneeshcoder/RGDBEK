function [x_values, RSE_propagation, iter] = RGDBEK(A, b, x0, max_iters, eta, tol)
    if nargin < 6, tol = 1e-6; end
    if nargin < 5, eta = 0.5; end
    if nargin < 4, max_iters = 400000; end

    [m, n] = size(A);
    x = x0;
    z = b;
    iter = 0;
    x_values = {x0};

    initRSE = norm(A*x-b)^2/norm(b)^2;
    RSE_propagation = {initRSE};

    block_size_col = ceil(eta * n);
    block_size_row = ceil(eta * m);

    while iter < max_iters
        iter = iter + 1;
        
        %% ---- Column Sampling ----
        col_scores = zeros(n,1);
        for j = 1:n
            aj = A(:,j);
            col_scores(j) = (abs(aj' * z)^2) / (norm(aj)^2 + eps);
            % col_scores(j) = (abs(aj' * z)^2) * (norm(aj)^2);
        end

        col_probs = col_scores / sum(col_scores);
        U_k = randsample(n, block_size_col, true, col_probs);
        A_Uk = A(:, U_k);
        
        % z update using lsqr
        z_new = z - A_Uk * lsqr(A_Uk, z);

        %% ---- Row Sampling ----
        row_scores = zeros(m,1);
        Ax = A * x;
        for i = 1:m
            ai = A(i,:);
            residual = b(i) - z_new(i) - Ax(i);
            row_scores(i) = (abs(residual)^2) / (norm(ai)^2 + eps);
            % row_scores(i) = (abs(residual)^2) * (norm(ai)^2);
        end

        row_probs = row_scores / sum(row_scores);
        J_k = randsample(m, block_size_row, true, row_probs);
        A_Jk = A(J_k, :);
        rhs = b(J_k) - z_new(J_k) - A_Jk * x;

        % x update using lsqr
        x_new = x + lsqr(A_Jk, rhs);

        %% ---- Check convergence ----
        x = x_new;
        z = z_new;
        x_values{end+1} = x;
        
        RSE = norm(A*x-b)^2/norm(b)^2;
        RSE_propagation{end+1} = RSE;

        % fprintf('Iter: %d, RSE: %f\n',iter, RSE);

        if RSE < tol
            break;
        end
    end
end
