import numpy as np

def rSVD(X, k, p=10, q=2):
    m, n = X.shape
    Omega = np.random.normal(size=(n, k + p)) # Controllato da seed globale
    Y = X @ Omega
    for _ in range(q):
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    U_hat, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat
    return U[:, :k], Sigma[:k], Vt[:k, :]
