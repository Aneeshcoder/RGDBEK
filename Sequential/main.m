clear all
clc

m = 500;
n = 8000;

sparse = true;
num_runs = 10;
run(m, n, sparse, num_runs)

%% Run Matrix
function [] = run(m, n, sparse, num_runs)
    if sparse
        sparse_matrix = sprandn(m,n,0.01);
        matrix = full(sparse_matrix);
    else
        matrix = randn(m, n);
    end
    
    A = matrix;    
    avgtime = 0;
    avgiter = 0;

    for i = 1:num_runs
        %% Initialization
        x_star = randn(size(A, 2), 1);
        transA = A';
        b = A * x_star + transA(1,:)';
        x0 = zeros(size(A, 2), 1);

        %% Compute RGDBEK
        tic;
        [x_values, res, iter] = RGDBEK(A, b, x0);
        time = toc;
        avgiter = avgiter + iter;
        avgtime = avgtime + time;
    end

    avgtime = avgtime / num_runs;
    avgiter = avgiter / num_runs;
    fprintf('Average Time: %f seconds\n', avgtime);
    fprintf('Average Iterations: %f\n\n', avgiter);
end
