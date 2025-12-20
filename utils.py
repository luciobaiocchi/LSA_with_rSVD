import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from constants import RANDOM_SEED


# Funzione analizza_miglior_k (definita qui per rendere lo script autonomo)
def analizza_miglior_k(X, k_min=3, k_max=10):
    print(f"Calcolo metriche per k da {k_min} a {k_max}...")
    silhouettes = []
    K_range = range(k_min, k_max + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(X)
        sil_score = silhouette_score(X, labels, random_state=RANDOM_SEED)
        silhouettes.append(sil_score)
        print(f"k={k}: Silhouette={sil_score:.4f}")

    best_idx = np.argmax(silhouettes)
    best_k = K_range[best_idx]

    # Plot rapido per report
    plt.figure(figsize=(8, 4))
    plt.plot(K_range, silhouettes, 'rs-', linewidth=2, markersize=8)
    plt.title('Silhouette Score (Separazione)')
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

    return best_k