import numpy as np
from scipy.sparse.linalg import svds
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import time

# 1. PREPARAZIONE DATI
print("Caricamento dati...")
# Usiamo subset='train' per fare prima nei test, usa 'all' per il report finale
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
print(data)
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
A = vectorizer.fit_transform(data.data)
print(f"Dimensione matrice: {A.shape} (Documenti x Parole)")
print(f"Sparsità: {A.nnz / (A.shape[0] * A.shape[1]):.4%}")

def rSVD(X, k, p=10, q=2): # Aggiunto parametro q (Power Iterations)
    """
    q: numero di power iterations (es. q=1 o q=2 sono sufficienti per NLP)
    """
    m, n = X.shape
    # Step 1: Matrice Test Random
    Omega = np.random.normal(size=(n, k + p))

    # Step 2: Campionamento con Power Iteration
    # Y = (A * A.T)^q * A * Omega
    # Questo serve a far convergere i vettori verso i valori singolari maggiori
    Y = X @ Omega
    for _ in range(q):
        Y = X @ (X.T @ Y) # Moltiplichiamo avanti e indietro

    # Step 3: Base Ortonormale (QR)
    Q, _ = np.linalg.qr(Y)

    # Step 4: Proiezione
    B = Q.T @ X

    # Step 5: SVD su B
    U_hat, Sigma, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Ricostruzione
    U = Q @ U_hat

    return U[:, :k], Sigma[:k], Vt[:k, :]

# 3. BENCHMARK
k = 50

print(f"\nConfronto per k={k}...")

# --- Metodo Classico (ARPACK via svds) ---
start = time.time()
U_c, S_c, Vt_c = svds(A, k=k)
time_classic = time.time() - start
# IMPORTANTE: svds restituisce i valori in ordine crescente! Invertiamoli.
S_c = S_c[::-1]
Vt_c = Vt_c[::-1, :]
U_c = U_c[:, ::-1]
print(f"Tempo SVD Classica: {time_classic:.4f} s")

# --- Metodo Randomizzato (rSVD) ---
start = time.time()
U_r, S_r, Vt_r = rSVD(A, k=k, p=20, q=2)
#U_r, S_r, Vt_r = rSVD(A, k=k, p=20) # p=20 per maggiore stabilità
time_random = time.time() - start
print(f"Tempo rSVD:        {time_random:.4f} s")
print(f"Speedup:           {time_classic / time_random:.2f}x")

# 4. ANALISI ERRORI E GRAFICI
# Calcolo Errore approssimazione sui Valori Singolari
error_sigma = np.linalg.norm(S_c - S_r) / np.linalg.norm(S_c)
print(f"Errore relativo sui Valori Singolari: {error_sigma:.6e}")

# Grafico 1: Scree Plot (Confronto Valori Singolari)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(S_c, 'k-', label='Exact (ARPACK)', linewidth=2)
plt.plot(S_r, 'r--', label='Randomized', linewidth=2)
plt.title(f'Decadimento Valori Singolari (k={k})')
plt.xlabel('Componente')
plt.ylabel('Valore Singolare')
plt.legend()
plt.grid(True)

# Grafico 2: Sparsità (usando markersize piccolo per non impallare il pc)
plt.subplot(1, 2, 2)
plt.spy(A, markersize=0.05, aspect='auto')
plt.title('Sparsità Matrice Originale')

plt.tight_layout()
plt.show()

# 5. TOPIC MODELING (Interpretabilità)
feature_names = vectorizer.get_feature_names_out()

print("\n--- Analisi Semantica (Top Words) ---")
# Confrontiamo il Topic 1 (Dominante) dei due metodi
def get_top_words(vec, n_words=8):
    top_indices = vec.argsort()[-n_words:][::-1]
    return [feature_names[i] for i in top_indices]

print(f"Topic 1 (Classica):   {get_top_words(Vt_c[0, :])}")
print(f"Topic 1 (Randomized): {get_top_words(Vt_r[0, :])}")

# Se i topic sono uguali (o molto simili), l'algoritmo funziona!
