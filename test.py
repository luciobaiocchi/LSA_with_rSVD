import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from stemming import stemmer, stemmed_tokenizer

# --- CONFIGURAZIONE RIPRODUCIBILITÀ ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



# 1. PREPARAZIONE DATI
print("Caricamento dati...")
# data: È una semplice lista di stringhe. Ogni elemento è un'email o un articolo del newsgroup.
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Creating custom stop word list
extra_words = ['don', 'just', 'like', 'know', 'people', 'think', 'does', 'use', 'good', 'time', 've']
final_stop_words = list(ENGLISH_STOP_WORDS) + extra_words

# Applichiamo lo stemming a tutte le stop words
stemmed_stop_words = [stemmer.stem(word) for word in final_stop_words]
# Rimuoviamo duplicati (perché diverse forme potrebbero diventare la stessa radice)
stemmed_stop_words = list(set(stemmed_stop_words))


#vectorizer: È il "dizionario". Sa che la parola "Honda" corrisponde alla colonna 450 e "Civic" alla colonna 920.
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words=stemmed_stop_words,
    tokenizer=stemmed_tokenizer,
    token_pattern=None,
    max_df=0.5
)

'''
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words=final_stop_words,
    max_df=0.5
)
'''

A = vectorizer.fit_transform(data.data)
#print(f"SHAPE {len(data.data[1])}")
print(f"Dimensione matrice: {A.shape} (Documenti x Parole)")
print(f"Sparsità: {A.nnz / (A.shape[0] * A.shape[1]):.4%}")

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

# 3. BENCHMARK
k = 50
print(f"\nEsecuzione rSVD per k={k}...")

# --- Metodo Randomizzato (rSVD) ---
start = time.time()


# U_r (18.846 , 50) Le coordinate di ogni documento nello spazio ridotto a 50 dimensioni
# S_r (50, 1) La Forza dei Concetti
# Vt_r (50, 10.000) La Riga 0 (Topic 1) ti dice quali parole compongono quel topic.

U_r, S_r, Vt_r = rSVD(A, k=k, p=20, q=10)
time_random = time.time() - start
print(f"Tempo rSVD: {time_random:.4f} s")

# 5. TOPIC MODELING & CLUSTERING
feature_names = vectorizer.get_feature_names_out()

# Preparazione dati ridotti
X_reduced = normalize(U_r)

# Trova il miglior K
print("\n--- Ricerca Best K ---")
best_n = analizza_miglior_k(X_reduced, k_min=5, k_max=10)
print(f"✅ Miglior numero di cluster identificato: {best_n}")

print("\n--- CLUSTERING AUTOMATICO (K-Means su rSVD) ---")
kmeans = KMeans(n_clusters=best_n, random_state=RANDOM_SEED, n_init=10)
labels = kmeans.fit_predict(X_reduced)

# Calcolo centroidi nello spazio originale per interpretazione
original_space_centroids = kmeans.cluster_centers_ @ Vt_r

# --- NUOVA SEZIONE: VISUALIZZAZIONE BARPLOT ---
print("\n--- Visualizzazione Parole Chiave per Cluster (Barplots) ---")

def plot_clusters_barchart(centroids, feature_names, n_top_words=6):
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

# Chiamata alla funzione di visualizzazione
plot_clusters_barchart(original_space_centroids, feature_names)

# Stampa esempi testuali (per verifica rapida contenuto)
print("\n--- Verifica Esempi Reali ---")
for i in range(best_n):
    docs_in_cluster = np.where(labels == i)[0]
    if len(docs_in_cluster) > 0:
        preview = data.data[docs_in_cluster[0]][:80].replace('\n', ' ')
        print(f"Cluster {i} Sample: {preview}...")


