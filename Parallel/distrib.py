from mpi4py import MPI
import numpy as np
import time
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_kernels = """
__global__ void coo_spmv_kernel(const int *row, const int *col, const float *val, 
                                const float *x, float *y, int nnz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nnz) {
        atomicAdd(&y[row[tid]], val[tid] * x[col[tid]]);
    }
}

__global__ void ell_spmv_kernel(const int *col_idx, const float *val, 
                                const float *x, float *y, 
                                int max_nnz, int row_offset, int rows) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int j = 0; j < max_nnz; j++) {
            int idx = row * max_nnz + j;
            int col = col_idx[idx];
            float v = val[idx];
            if (col >= 0) sum += v * x[col];
        }
        y[row_offset + row] = sum;
    }
}

__global__ void csr_spmv_kernel(const int *indptr, const int *indices, const float *data,
                                const float *x, float *y, int row_offset, int rows) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) {
        float sum = 0.0f;
        int start = indptr[row];
        int end = indptr[row + 1];
        for (int j = start; j < end; ++j) {
            sum += data[j] * x[indices[j]];
        }
        y[row_offset + row] = sum;
    }
}

__global__ void hyb_spmv_ell(const int *col_idx, const float *val, 
                             const float *x, float *y, 
                             int max_nnz, int row_offset, int rows) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int j = 0; j < max_nnz; j++) {
            int idx = row * max_nnz + j;
            int col = col_idx[idx];
            float v = val[idx];
            if (col >= 0) sum += v * x[col];
        }
        atomicAdd(&y[row_offset + row], sum);
    }
}

__global__ void hyb_spmv_coo(const int *row, const int *col, const float *val, 
                             const float *x, float *y, int nnz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nnz) {
        atomicAdd(&y[row[tid]], val[tid] * x[col[tid]]);
    }
}
"""

mod = SourceModule(cuda_kernels)
coo_kernel = mod.get_function("coo_spmv_kernel")
ell_kernel = mod.get_function("ell_spmv_kernel")
csr_kernel = mod.get_function("csr_spmv_kernel")
hyb_ell_kernel = mod.get_function("hyb_spmv_ell")
hyb_coo_kernel = mod.get_function("hyb_spmv_coo")

def partition_matrix_bsas(matrix, target_blocks=32):
    m = matrix.shape[0]
    nnz_per_row = np.diff(matrix.indptr)
    segment_size = int(np.ceil(np.sum(nnz_per_row) / target_blocks))
    blocks, current, running_sum = [], [], 0
    for i, nz in enumerate(nnz_per_row):
        current.append(i)
        running_sum += nz
        if running_sum >= segment_size:
            blocks.append(list(current))
            current, running_sum = [], 0
    if current: blocks.append(list(current))
    return blocks

def choose_block_format(submat, q_thresh=2e-5, tl=0.2, tu=1.1):
    x, y = submat.shape
    nnz = submat.nnz
    q = nnz / (x * y)
    nnz_per_row = np.diff(submat.indptr)
    ARL = float(nnz) / x if x > 0 else 0.0
    CV = np.std(nnz_per_row) / (ARL + 1e-12)
    if q < q_thresh:
        return 'COO', submat
    elif CV < tl:
        return 'ELL', submat
    elif CV > tu:
        return 'HYB', submat
    else:
        return 'CSR', submat

class BSASBlock:
    def __init__(self, matrix, row_indices, format_params=None):
        submat = matrix[row_indices, :]
        if not isinstance(submat, csr_matrix):
            submat = submat.tocsr()
        self.block_format, self.submat = choose_block_format(submat, **(format_params or {}))
        self.row_offset = row_indices[0]
        self.rows = len(row_indices)
        if self.block_format == 'COO':
            coo = self.submat.tocoo()
            self.nnz = coo.nnz
            if self.nnz > 0:
                self.gpu_row = cuda.to_device((coo.row + self.row_offset).astype(np.int32))
                self.gpu_col = cuda.to_device(coo.col.astype(np.int32))
                self.gpu_val = cuda.to_device(coo.data.astype(np.float32))
            else:
                self.gpu_row = None
                self.gpu_col = None
                self.gpu_val = None
        elif self.block_format == 'ELL':
            nnz_per_row = np.diff(self.submat.indptr)
            max_nnz = max(nnz_per_row) if len(nnz_per_row) > 0 else 0
            ell_val = np.zeros((self.rows, max_nnz), dtype=np.float32)
            ell_col = -np.ones((self.rows, max_nnz), dtype=np.int32)
            for i in range(self.rows):
                start, end = self.submat.indptr[i], self.submat.indptr[i+1]
                col_ = self.submat.indices[start:end]
                val_ = self.submat.data[start:end]
                ell_val[i, :len(val_)] = val_
                ell_col[i, :len(col_)] = col_
            self.max_nnz = max_nnz
            self.gpu_ell_val = cuda.to_device(ell_val)
            self.gpu_ell_col = cuda.to_device(ell_col)
        elif self.block_format == 'CSR':
            indptr = self.submat.indptr.astype(np.int32)
            indices = self.submat.indices.astype(np.int32)
            data = self.submat.data.astype(np.float32)
            self.nrows = self.submat.shape[0]
            self.gpu_indptr = cuda.to_device(indptr)
            self.gpu_indices = cuda.to_device(indices)
            self.gpu_data = cuda.to_device(data)
        elif self.block_format == 'HYB':
            nnz_per_row = np.diff(self.submat.indptr)
            V = max(min(int(np.mean(nnz_per_row)), int(np.median(nnz_per_row))), 1)
            nrows = self.submat.shape[0]
            ell_val = np.zeros((nrows, V), dtype=np.float32)
            ell_col = -np.ones((nrows, V), dtype=np.int32)
            coo_rows, coo_cols, coo_vals = [], [], []
            for i in range(nrows):
                start, end = self.submat.indptr[i], self.submat.indptr[i+1]
                nnz_in_row = end - start
                n_ell = min(V, nnz_in_row)
                if n_ell > 0:
                    ell_val[i, :n_ell] = self.submat.data[start:start+n_ell]
                    ell_col[i, :n_ell] = self.submat.indices[start:start+n_ell]
                if nnz_in_row > V:
                    coo_rows.extend([i]* (nnz_in_row - V))
                    coo_cols.extend(self.submat.indices[start+V:end])
                    coo_vals.extend(self.submat.data[start+V:end])
            self.hyb_V = V
            self.gpu_hyb_ell_val = cuda.to_device(ell_val)
            self.gpu_hyb_ell_col = cuda.to_device(ell_col)
            self.hyb_nrows = nrows
            if coo_rows:
                coo_rows = np.array(coo_rows, dtype=np.int32) + self.row_offset
                self.gpu_hyb_coo_row = cuda.to_device(coo_rows)
                self.gpu_hyb_coo_col = cuda.to_device(np.array(coo_cols, dtype=np.int32))
                self.gpu_hyb_coo_val = cuda.to_device(np.array(coo_vals, dtype=np.float32))
                self.hyb_coo_nnz = len(coo_rows)
            else:
                self.gpu_hyb_coo_row = None
                self.hyb_coo_nnz = 0
        else:
            raise NotImplementedError("Invalid format for GPU SpMV.")

    def spmv(self, gpu_x, gpu_y):
        if self.block_format == 'COO':
            if self.nnz == 0 or self.gpu_row is None:
                return
            block = 256
            grid = int((self.nnz + block - 1) / block)
            coo_kernel(self.gpu_row, self.gpu_col, self.gpu_val, gpu_x, gpu_y,
                       np.int32(self.nnz), block=(block,1,1), grid=(grid,1))
        elif self.block_format == 'ELL':
            block = 256
            grid = int((self.rows + block - 1) / block)
            ell_kernel(self.gpu_ell_col, self.gpu_ell_val, gpu_x, gpu_y,
                       np.int32(self.max_nnz), np.int32(self.row_offset), np.int32(self.rows),
                       block=(block,1,1), grid=(grid,1))
        elif self.block_format == 'CSR':
            block = 256
            grid = int((self.nrows + block - 1) / block)
            csr_kernel(self.gpu_indptr, self.gpu_indices, self.gpu_data, gpu_x, gpu_y,
                       np.int32(self.row_offset), np.int32(self.nrows),
                       block=(block,1,1), grid=(grid,1))
        elif self.block_format == 'HYB':
            block = 256
            grid = int((self.hyb_nrows + block - 1) / block)
            hyb_ell_kernel(self.gpu_hyb_ell_col, self.gpu_hyb_ell_val, gpu_x, gpu_y,
                           np.int32(self.hyb_V), np.int32(self.row_offset), np.int32(self.hyb_nrows),
                           block=(block,1,1), grid=(grid,1))
            if self.hyb_coo_nnz > 0:
                grid2 = int((self.hyb_coo_nnz + block - 1) / block)
                hyb_coo_kernel(self.gpu_hyb_coo_row, self.gpu_hyb_coo_col, self.gpu_hyb_coo_val,
                               gpu_x, gpu_y, np.int32(self.hyb_coo_nnz),
                               block=(block,1,1), grid=(grid2,1))
        else:
            raise NotImplementedError("Block type not supported in GPU execution!")

def bsas_spmv(matrix_csr, vector, target_blocks=32):
    assert isinstance(matrix_csr, csr_matrix) and vector.ndim == 1
    n_rows = matrix_csr.shape[0]
    blocks = partition_matrix_bsas(matrix_csr, target_blocks)
    bsas_blocks = [BSASBlock(matrix_csr, block) for block in blocks]
    gpu_x = cuda.to_device(vector.astype(np.float32))
    gpu_y = cuda.to_device(np.zeros(n_rows, dtype=np.float32))
    for block in bsas_blocks:
        block.spmv(gpu_x, gpu_y)
    result = np.zeros(n_rows, dtype=np.float32)
    cuda.memcpy_dtoh(result, gpu_y)
    return result

def rgdbek():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parameters
    m_total = 50000         # rows in A
    n = 50000               # columns in A
    max_iter = 500
    tol = 1e-4          
    eta = 0.1           
    density = 0.01
    d = m_total // size      

    # Determine rows per rank
    rows_per_rank = [m_total // size] * size
    for i in range(m_total % size):
        rows_per_rank[i] += 1
    start_row = sum(rows_per_rank[:rank])
    # end_row = start_row + rows_per_rank[rank]

    if rank == 0:
        A_global = sp.rand(m_total, n, density=density, format='csr').astype(np.float32)
        for r in range(size):
            indices = np.arange(sum(rows_per_rank[:r]), sum(rows_per_rank[:r + 1]))
            if r == 0:
                local_matrix = A_global[indices, :]
            else:
                submat = A_global[indices, :]
                comm.send((submat.data, submat.indices, submat.indptr, submat.shape), dest=r)
    else:
        data, indices, indptr, shape = comm.recv(source=0)
        local_matrix = sp.csr_matrix((data, indices, indptr), shape=shape)

    A_local = local_matrix
    x_true = np.random.rand(n).astype(np.float32) if rank == 0 else None
    x_true = comm.bcast(x_true, root=0)
    b_local = bsas_spmv(A_local, x_true)

    x = np.zeros(n, dtype=np.float32)
    z = b_local.copy()

    # Precompute column and row norms for probability calculations
    At_norm_sq = np.array(A_local.power(2).sum(axis=0)).ravel()
    Ar_norm_sq = np.array(A_local.power(2).sum(axis=1)).ravel()

    comm.Barrier()
    start_time = time.time()

    num_iter = 0
    for k in range(max_iter):
        # === PHASE 1: COLUMN BLOCK SAMPLING ===
        Az = bsas_spmv(A_local.transpose().tocsr(), z)
        
        # Compute column probabilities
        eps_j = np.abs(Az)**2 / (At_norm_sq + 1e-20)

        # Gather probabilities across all processors
        all_eps_j = comm.allgather(eps_j)
        global_eps_j = np.sum(all_eps_j, axis=0) + 1e-20
        P_col = global_eps_j / (global_eps_j.sum())
        
        n_eta = max(1, int(n * eta))
        sampled_cols = np.random.choice(n, size=n_eta, replace=False, p=P_col)
        
        # Solve least squares for column submatrix
        A_Uk = A_local[:, sampled_cols]
        zsol = lsqr(A_Uk, z)[0]
        z_update = bsas_spmv(A_Uk, zsol)
        z = z - z_update
        
        # === PHASE 2: ROW BLOCK SAMPLING ===
        Ax = bsas_spmv(A_local, x)
        residual = b_local - z - Ax
        
        # Compute row probabilities
        eps_i = np.abs(residual)**2 / (Ar_norm_sq + 1e-20)
        P_row = eps_i / (eps_i.sum())

        m_eta = max(1, int(d * eta))
        sampled_rows = np.random.choice(d, size=m_eta, replace=False, p=P_row)
        
        # Solve least squares for row submatrix
        A_Jk = A_local[sampled_rows, :]
        rhs = b_local[sampled_rows] - z[sampled_rows] - bsas_spmv(A_Jk, x)        
        x_updatelocal = lsqr(A_Jk, rhs)[0]
        
        x_global = np.zeros(n, dtype=x_updatelocal.dtype)
        comm.Allreduce(x_updatelocal, x_global, op=MPI.SUM)
        x_update = x_global / size

        x_update = x_update.astype(x.dtype)
        x = x + x_update

        if k%1 == 0:
            # === CONVERGENCE CHECK ===
            final_residual = b_local - bsas_spmv(A_local, x)
            local_residual_sq = np.linalg.norm(final_residual)**2
            local_b_sq = np.linalg.norm(b_local)**2

            # Global reduction
            global_residual_sq = comm.allreduce(local_residual_sq, op=MPI.SUM)
            global_b_sq = comm.allreduce(local_b_sq, op=MPI.SUM)
            
            # RSE Calculation
            RSE = global_residual_sq / (global_b_sq)
            
            # if rank == 0:
            #     print(f"Iteration {k+1} Relative Squared Error (RSE): {RSE:.6e}")
            
            if RSE < tol:
                num_iter = k + 1
                break

    comm.Barrier()
    end_time = time.time()
    runtime = end_time - start_time

    if rank == 0:
        print(f"Converged in {num_iter} iterations")
        print(f"Total time taken: {runtime:.4f} seconds")
        print(f"RSE: {RSE:.6e}")

if __name__ == "__main__":
    rgdbek()
