import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from stemming import stemmed_tokenizer, stemmer
from utils import plot_clusters_barchart


class SemanticEngine:
    def __init__(self, n_topics=50, n_clusters=10):
        self.n_topics = n_topics
        self.n_clusters = n_clusters
        self.vectorizer = None
        self.U = None
        self.Vt = None
        self.kmeans = None
        self.data_raw = None
        self.is_fitted = False

    def fit(self, data, extra_stop_words=[]):
        """Addestra l'intera pipeline."""
        self.data_raw = data
        print("1. Preparazione Stop Words e Vectorizer (con N-Grams)...")
        
        # Stop words potenziate
        final_stop_words = list(ENGLISH_STOP_WORDS) + extra_stop_words
        stemmed_stops = list(set([stemmer.stem(w) for w in final_stop_words]))
        
        # Aggiunta N-GRAMS (1,2)
        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words=stemmed_stops,
            tokenizer=stemmed_tokenizer,
            token_pattern=None,
            ngram_range=(1, 2), # <--- CATTURA "social_network", "operating_system"
            max_df=0.5,
            min_df=5 # Ignora termini che appaiono in meno di 5 documenti (rimuove typo rari)
        )
        
        A = self.vectorizer.fit_transform(data)
        
        print("2. Decomposizione rSVD...")
        # Importiamo rSVD qui o usiamo self se la integri nella classe
        from functions import rSVD 
        self.U, _, self.Vt = rSVD(A, k=self.n_topics)
        
        # Normalizzazione per similarità coseno
        self.U = normalize(self.U)
        
        print(f"3. Clustering (k={self.n_clusters})...")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.U)
        self.is_fitted = True
        print("✅ Addestramento completato.")

    def get_cluster_keywords(self, cluster_id, n_words=10):
        """Restituisce le parole chiave per un dato cluster."""
        centroid = self.kmeans.cluster_centers_[cluster_id]
        # Proiettiamo il centroide indietro nello spazio delle parole: Centroide * Vt
        original_space_centroid = centroid @ self.Vt
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = original_space_centroid.argsort()[-n_words:][::-1]
        return [feature_names[i] for i in top_indices]
    
    def plot_barchart_clusters(self):
        feature_names = self.vectorizer.get_feature_names_out()
        original_space_centroids = self.kmeans.cluster_centers_ @ self.Vt
        plot_clusters_barchart(original_space_centroids, feature_names)

    def search(self, query, top_k=5):
        """Motore di ricerca semantico."""
        if not self.is_fitted:
            raise Exception("Il modello non è stato addestrato! Esegui .fit()")
            
        # 1. Trasforma query in vettore parole
        query_vec = self.vectorizer.transform([query])
        
        # 2. Proietta query nello spazio Topic (usando Vt)
        # Nota: La proiezione corretta è Query * Vt.T
        query_topic = query_vec @ self.Vt.T
        query_topic = normalize(query_topic) # Importante normalizzare
        
        # 3. Similarità Coseno con tutti i documenti
        sims = cosine_similarity(query_topic, self.U).flatten()
        
        # 4. Top K results
        top_indices = sims.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'id': idx,
                'score': sims[idx],
                'cluster': self.labels[idx],
                'text': self.data_raw[idx] + "..."
            })
        return results