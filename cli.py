import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
# Assicurati che i file stemming.py, functions.py e utils.py siano nella stessa cartella
from stemming import stemmer, stemmed_tokenizer
from functions import rSVD
from utils import analizza_miglior_k, plot_clusters_barchart
from semantic_engine import SemanticEngine

# --- CONFIGURAZIONE RIPRODUCIBILITÀ ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Nome del file locale dove salvare il dataset
DATASET_FILE = "dataset_20newsgroups.pkl"

# 1. PREPARAZIONE DATI
print("Gestione Dataset...")

if os.path.exists(DATASET_FILE):
    print(f"Trovato dataset locale '{DATASET_FILE}'. Caricamento in corso...")
    with open(DATASET_FILE, 'rb') as f:
        data = pickle.load(f)
    print("Dataset caricato correttamente.")
else:
    print(f"Dataset locale non trovato. Scaricamento da scikit-learn in corso...")
    # data: È una semplice lista di stringhe. Ogni elemento è un'email o un articolo del newsgroup.
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    print(f"Salvataggio dataset in '{DATASET_FILE}' per uso futuro...")
    with open(DATASET_FILE, 'wb') as f:
        pickle.dump(data, f)
    print("Salvataggio completato.")

# Creating custom stop word list
extra_words = ['don', 'just', 'like', 'know', 'people', 'think', 'does', 'use', 'good', 'time', 've', 'make', 'say','thank', 'advanc', 'hi', 'appreci']
final_stop_words = list(ENGLISH_STOP_WORDS) + extra_words

# Applichiamo lo stemming a tutte le stop words
# Nota: Questo può generare un warning in scikit-learn perché le parole
# vengono trasformate (es. "anywhere" -> "anywh") ma è il comportamento corretto qui.
#stemmed_stop_words = [stemmer.stem(word) for word in final_stop_words]
# Rimuoviamo duplicati
#stemmed_stop_words = list(set(stemmed_stop_words))

engine = SemanticEngine(n_topics=50, n_clusters=8)
    
# Addestra
#extra_words = ['don', 'just', 'like', 'know', 'people']
engine.fit(data.data, extra_words)

# 1. Analisi Cluster
print("\n--- Analisi Cluster ---")
for i in range(8):
    kw = engine.get_cluster_keywords(i)
    print(f"Cluster {i}: {', '.join(kw)}")
    
# 2. PROVA IL MOTORE DI RICERCA
print("\n--- Test Search Engine ---")
while True:
    user_input = input("Inserisci una query (o più separate da virgola): ")
    queries = [q.strip() for q in user_input.split(',')]

    for q in queries:
        if not q: continue # Salta stringhe vuote
        print(f"\n🔎 Query: '{q}'")
        
        res = engine.search(q, top_k=5)
        
        for r in res:
            # CORREZIONE QUI:
            # Calcoliamo il testo pulito PRIMA di stamparlo
            #clean_text = r['text'].replace('\n', ' ')
            
            # Ora usiamo la variabile clean_text nella f-string senza backslash
            print("=========================================================================")
            print(f"   [{r['score']:.4f}] (Cluster {r['cluster']}): {r['text']}")

'''
print("Vettorizzazione in corso (TF-IDF)...")
# vectorizer: È il "dizionario". Sa che la parola "Honda" corrisponde alla colonna X.
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    stop_words=stemmed_stop_words,
    tokenizer=stemmed_tokenizer,
    token_pattern=None, # Importante per evitare warning quando si usa un tokenizer custom
    max_df=0.5
)

A = vectorizer.fit_transform(data.data)

print(f"Dimensione matrice: {A.shape} (Documenti x Parole)")
print(f"Sparsità: {A.nnz / (A.shape[0] * A.shape[1]):.4%}")

# 3. BENCHMARK
k = 50
print(f"\nEsecuzione rSVD per k={k}...")

# U_r (18.846 , 50): Coordinate documenti nello spazio ridotto
# S_r (50, 1): Importanza dei topic
# Vt_r (50, 10.000): Relazione Topic-Parole

start = time.time()
U_r, S_r, Vt_r = rSVD(A, k=k, p=20, q=2) # q=2 è solitamente sufficiente e più veloce di 10
time_random = time.time() - start
print(f"Tempo rSVD: {time_random:.4f} s")

# 5. TOPIC MODELING & CLUSTERING
feature_names = vectorizer.get_feature_names_out()

# Preparazione dati ridotti
X_reduced = normalize(U_r)

# Trova il miglior K
print("\n--- Ricerca Best K ---")
# Ampliato leggermente il range per vedere meglio l'andamento
best_n = analizza_miglior_k(X_reduced, k_min=5, k_max=12)
print(f"✅ Miglior numero di cluster identificato: {best_n}")

print("\n--- CLUSTERING AUTOMATICO (K-Means su rSVD) ---")
kmeans = KMeans(n_clusters=best_n, random_state=RANDOM_SEED, n_init=10)
labels = kmeans.fit_predict(X_reduced)

# Calcolo centroidi nello spazio originale per interpretazione
original_space_centroids = kmeans.cluster_centers_ @ Vt_r

# --- VISUALIZZAZIONE BARPLOT ---
print("Generazione grafici...")
plot_clusters_barchart(original_space_centroids, feature_names)

# Stampa esempi testuali (per verifica rapida contenuto)
print("\n--- Verifica Esempi Reali ---")
for i in range(best_n):
    docs_in_cluster = np.where(labels == i)[0]
    if len(docs_in_cluster) > 0:
        # Prendi il primo documento e pulisci i newline per la stampa
        preview = data.data[docs_in_cluster[0]][:80].replace('\n', ' ')
        print(f"Cluster {i} Sample: {preview}...")
'''