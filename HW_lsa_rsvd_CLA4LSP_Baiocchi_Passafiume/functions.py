import numpy as np
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def rSVD(X, k, p=10, q=2):
    # X m,n
    # Om n, k+p
    # Y m, k+p 
    m, n = X.shape
    Omega = np.random.normal(size=(n, k + p)) # P = oversampling, increase drastically the aìprobability of extracting usefull data
    Y = X @ Omega
    for _ in range(q):   #POWER ITERATIONS, increase big eigenvals, decrease small eigenvals
        Y = X @ (X.T @ Y)
    Q, _ = np.linalg.qr(Y)
    # Q m, k+p
    B = Q.T @ X   # B is the projection of X in the smaller space ->  dim :(k + p ), n
    # B k+p, n
    U_hat, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    # Sigma k+p, k+p
    # U_hat k+p, k+p
    # Vt k+p, n
    U = Q @ U_hat # Back to the bigger space
    # U m, k+p
    return U[:, :k], Sigma[:k], Vt[:k, :]       #We return without oversampling col/rows  
    # U m, k
    # Sigma k, k
    # Vt k, n