import os
import pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from semantic_engine import SemanticEngine

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
DATASET_FILE = "dataset_20newsgroups.pkl"

if os.path.exists(DATASET_FILE):
    print(f"Trovato dataset locale '{DATASET_FILE}'. Caricamento in corso...")
    with open(DATASET_FILE, 'rb') as f:
        data = pickle.load(f)
    print("Dataset caricato correttamente.")
else:
    print(f"Dataset locale non trovato. Scaricamento da scikit-learn in corso...")
    # data: String lis, every item = mail
    data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    print(f"Salvataggio dataset in '{DATASET_FILE}' per uso futuro...")
    with open(DATASET_FILE, 'wb') as f:
        pickle.dump(data, f)
    print("Salvataggio completato.")

# custom stop word list
extra_words = ['don', 'just', 'like', 'know', 'people', 'think', 'does', 'use', 'good', 'time', 've', 'make', 'say','thank', 'advanc', 'hi', 'appreci', 'did']

engine = SemanticEngine(n_topics=50, n_clusters=8)
engine.fit(data.data, extra_words)

# CLUSTER LOG
print("\n--- Analisi Cluster ---")
for i in range(8):
    kw = engine.get_cluster_keywords(i)
    print(f"Cluster {i}: {', '.join(kw)}")

engine.plot_barchart_clusters()

# SEARCH ENGINE
print("\n--- Test Search Engine ---")
while True:
    user_input = input("Inserisci una query (o più separate da virgola): ")
    queries = [q.strip() for q in user_input.split(',')]

    for q in queries:
        if not q: continue # Salta stringhe vuote
        print(f"\n🔎 Query: '{q}'")
        
        res = engine.search(q, top_k=5)
        
        for r in res:
            print("=========================================================================")
            print(f"   [{r['score']:.4f}] (Cluster {r['cluster']}): {r['text']}")