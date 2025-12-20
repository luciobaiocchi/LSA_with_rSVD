import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from stemming import stemmer, stemmed_tokenizer
from functions import rSVD
from utils import analizza_miglior_k, plot_clusters_barchart

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

# 3. BENCHMARK
k = 50
print(f"\nEsecuzione rSVD per k={k}...")

# U_r (18.846 , 50) Le coordinate di ogni documento nello spazio ridotto a 50 dimensioni
# S_r (50, 1) La Forza dei Concetti
# Vt_r (50, 10.000) La Riga 0 (Topic 1) ti dice quali parole compongono quel topic.

start = time.time()
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

plot_clusters_barchart(original_space_centroids, feature_names)

# Stampa esempi testuali (per verifica rapida contenuto)
print("\n--- Verifica Esempi Reali ---")
for i in range(best_n):
    docs_in_cluster = np.where(labels == i)[0]
    if len(docs_in_cluster) > 0:
        preview = data.data[docs_in_cluster[0]][:80].replace('\n', ' ')
        print(f"Cluster {i} Sample: {preview}...")