import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def rgdbek(A, b, max_iter=500, tol=1e-4, eta=0.1):
    m, n = A.shape
    x = np.zeros(n, dtype=np.float32)
    z = b.copy()

    At_norm_sq = np.array(A.power(2).sum(axis=0)).ravel()
    Ar_norm_sq = np.array(A.power(2).sum(axis=1)).ravel()

    start_time = time.time()

    for k in range(max_iter):
        Az = A.transpose().dot(z)
        eps_j = np.abs(Az)**2 / (At_norm_sq + 1e-20)
        n_eta = max(1, int(n * eta))
        sampled_cols = np.random.choice(n, size=n_eta, replace=False, p=eps_j/eps_j.sum())
        A_Uk = A[:, sampled_cols]
        zsol = lsqr(A_Uk, z)[0]
        z_update = A_Uk.dot(zsol)
        z = z - z_update

        Ax = A.dot(x)
        residual = b - z - Ax
        eps_i = np.abs(residual)**2 / (Ar_norm_sq + 1e-20)
        P_row = eps_i / eps_i.sum()
        m_eta = max(1, int(m * eta))
        sampled_rows = np.random.choice(m, size=m_eta, replace=False, p=P_row)
        A_Jk = A[sampled_rows, :]
        rhs = b[sampled_rows] - z[sampled_rows] - A_Jk.dot(x)
        x_update = lsqr(A_Jk, rhs)[0]
        x = x + x_update

        final_residual = b - A.dot(x)
        RSE = np.linalg.norm(final_residual)**2 / np.linalg.norm(b)**2
        
        if (k + 1) % 500 == 0:
            print(f"Iteration {k+1}: RSE = {RSE:.6e}")
        
        if RSE < tol:
            print(f"Converged at iteration {k+1} with RSE {RSE:.6e}")
            break

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")
    return x

# 1. Generate mesh grid and triangulate
nx, ny = 25, 25
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T
tri = Delaunay(points)
elements = tri.simplices

def shape_func_derivatives(xy):
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    A = 0.5 * ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    B = np.vstack((b, c)) / (2 * A)
    return B, A

# 2. Assemble stiffness matrix A and load vector b
N = len(points)
A = np.zeros((N, N))
b_vec = np.zeros(N)

for elem in elements:
    xy = points[elem]
    B, Area = shape_func_derivatives(xy)
    Ke = Area * (B.T @ B)
    centroid = np.mean(xy, axis=0)
    f_val = 2 * np.pi**2 * np.sin(np.pi * centroid[0]) * np.sin(np.pi * centroid[1])

    Fe = f_val * Area / 3 * np.ones(3)
    for i_local, iglob in enumerate(elem):
        b_vec[iglob] += Fe[i_local]
        for j_local, jglob in enumerate(elem):
            A[iglob, jglob] += Ke[i_local, j_local]

# 3. Apply Dirichlet BC (u=0)
boundary_nodes = np.where(
    (np.isclose(points[:, 0], 0)) | (np.isclose(points[:, 0], 1)) |
    (np.isclose(points[:, 1], 0)) | (np.isclose(points[:, 1], 1)))[0]
for node in boundary_nodes:
    A[node, :] = 0
    A[:, node] = 0
    A[node, node] = 1
    b_vec[node] = 0

# 4. Solve linear system
print("Size of A:", A.shape)
A_sp = sp.csr_matrix(A)
U = rgdbek(A_sp, b_vec, max_iter=100000, tol=1e-6, eta=0.75)

# 5. Compute exact solution and error
U_exact = np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
error = np.abs(U - U_exact)

# Metrics
L2_error = np.linalg.norm(U - U_exact) / np.linalg.norm(U_exact)
print(f"L2 Relative Error: {L2_error:.6e}")
print(f"Standard Deviation of Error: {np.std(error):.6e}")
print(f"Condition Number of A: {np.linalg.cond(A):.6e}")
print(f"Sparsity of A: {100 * (1 - np.count_nonzero(A) / A.size):.2f}%")
print(f"Frobenius Norm of A: {np.linalg.norm(A, 'fro'):.6e}")
print(f"Number of elements: {len(elements)}")

def plot_solution_with_mesh(vals, title):
    fig, ax = plt.subplots(figsize=(6,5))
    triang = mtri.Triangulation(points[:, 0], points[:, 1], elements)
    tpc = ax.tripcolor(triang, vals, shading='gouraud', cmap='viridis')
    ax.triplot(triang, color='black', linewidth=0.75, alpha=1)

    x_ticks = np.linspace(points[:, 0].min(), points[:, 0].max(), num=11)
    y_ticks = np.linspace(points[:, 1].min(), points[:, 1].max(), num=11)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='both', which='major', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Times New Roman')

    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.tick_params(axis='both', which='major', labelsize=18)

    cbar = fig.colorbar(tpc, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname('Times New Roman')

    plt.tight_layout()
    return fig

fig1 = plot_solution_with_mesh(U, 'FEM Solution')
fig2 = plot_solution_with_mesh(U_exact, 'Exact Solution')
fig3 = plot_solution_with_mesh(error, 'Absolute Error')

fig1.savefig('poisson_FEM.png', dpi=600)
fig2.savefig('poisson_Exact.png', dpi=600)
fig3.savefig('poisson_Error.png', dpi=600)