import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from constants import RANDOM_SEED

def plot_clusters_barchart(centroids, feature_names, n_top_words=6):
    print("\n--- Visualizzazione Parole Chiave per Cluster (Barplots) ---")
    n_clusters = centroids.shape[0]

    # Calcoliamo righe e colonne per la griglia di grafici
    n_cols = 2
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), constrained_layout=True)
    axes = axes.flatten() # Appiattiamo per iterare facilmente

    # Colormap per dare colori diversi ai cluster
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        ax = axes[i]

        # Estrazione top words e pesi
        # argsort ordina crescente, prendiamo gli ultimi n_top_words e invertiamo
        top_indices = centroids[i].argsort()[-n_top_words:]
        top_words = [feature_names[j] for j in top_indices]
        top_weights = centroids[i][top_indices]

        # Creazione Barplot Orizzontale
        # Usiamo i dati così come sono (dal meno importante al più importante) per barh
        ax.barh(top_words, top_weights, color=colors[i % 10], alpha=0.8)

        ax.set_title(f"CLUSTER {i}", fontsize=12, fontweight='bold', color='black')
        ax.set_xlabel("Peso (Importanza)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Nascondiamo eventuali subplot vuoti (se n_clusters è dispari)
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Parole Chiave Top {n_top_words} per Cluster", fontsize=16, y=1.02)
    plt.show()

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