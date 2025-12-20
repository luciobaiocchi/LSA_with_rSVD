import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from semantic_engine import SemanticEngine
from sklearn.datasets import fetch_20newsgroups

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LSA Semantic Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 15px;
        border-left: 5px solid #4e8cff;
    }
    .result-score {
        font-weight: bold;
        color: #4e8cff;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 2px 5px;
        border-radius: 3px;
    }
    .stProgress .st-bo {
        background-color: #4e8cff;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
DATASET_FILE = "dataset_20newsgroups.pkl"
EXTRA_STOP_WORDS = ['don', 'just', 'like', 'know', 'people', 'think', 'does', 'use', 'good', 'time', 've', 'make', 'say','thank', 'advanc', 'hi', 'appreci']

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_data():
    """Load or download the dataset."""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'rb') as f:
            data = pickle.load(f)
    else:
        with st.spinner("Downloading 20 Newsgroups dataset..."):
            data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
            with open(DATASET_FILE, 'wb') as f:
                pickle.dump(data, f)
    return data

@st.cache_resource
def train_model(_data, n_topics=50, n_clusters=8):
    """Initialize and train the SemanticEngine."""
    engine = SemanticEngine(n_topics=n_topics, n_clusters=n_clusters)
    with st.spinner(f"Training LSA Model (Topics: {n_topics}, Clusters: {n_clusters})..."):
        engine.fit(_data.data, EXTRA_STOP_WORDS)
    return engine

# --- MAIN APP ---
def main():
    st.title("🧩 LSA Semantic Browser")
    st.markdown("Explore latent topics in the **20 Newsgroups** dataset using **Latent Semantic Analysis (rSVD)**.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        n_clusters = st.slider("Number of Clusters", min_value=3, max_value=20, value=8)
        n_topics = st.slider("Number of Topics (Dimensions)", min_value=10, max_value=100, value=50)
        
        if st.button("Reload/Retrain Model"):
            st.cache_resource.clear()
            st.rerun()
            
        st.markdown("---")
        st.info("Built with Streamlit & rSVD")

    # Load Data & Model
    data = load_data()
    engine = train_model(data, n_topics=n_topics, n_clusters=n_clusters)

    # Tabs for different views
    tab_search, tab_clusters, tab_dataset = st.tabs(["🔎 Semantic Search", "📊 Cluster Analysis", "📁 Dataset Preview"])

    # --- TAB 1: SEARCH ---
    with tab_search:
        st.subheader("Semantic Search Engine")
        query = st.text_input("Enter your query:", placeholder="e.g., space exploration, computer graphics, middle east conflict...")
        
        if query:
            results = engine.search(query, top_k=10)
            
            st.markdown(f"### Results for *'{query}'*")
            for res in results:
                # Format score as percentage bar
                score_pct = max(0, min(100, int(res['score'] * 100)))
                
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="result-score">Sim: {res['score']:.4f}</span>
                            <span style="background-color: #e0e0e0; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">Cluster {res['cluster']}</span>
                        </div>
                        <div style="margin-top: 10px; font-family: monospace; white-space: pre-wrap; max-height: 200px; overflow-y: auto;">
                            {res['text'][:1000]}{'...' if len(res['text']) > 1000 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # --- TAB 2: CLUSTERS ---
    with tab_clusters:
        st.subheader("Cluster Keywords Visualization")
        
        # Prepare data for all clusters
        cluster_data = []
        for i in range(n_clusters):
            kw = engine.get_cluster_keywords(i, n_words=10)
            # Since get_cluster_keywords only returns words, we need to get their weights manually if we want to plot them accurately.
            # But get_cluster_keywords in semantic_engine.py doesn't return weights.
            # Let's peek at semantic_engine.py again or just use rank/order.
            # Actually, let's reverse engineer weights from the generic plotting function or just mock them based on rank if needed.
            # Wait, I can use the same logic as get_cluster_keywords to get weights.
            
            centroid = engine.kmeans.cluster_centers_[i]
            original_space_centroid = centroid @ engine.Vt
            feature_names = engine.vectorizer.get_feature_names_out()
            top_indices = original_space_centroid.argsort()[-10:][::-1]
            
            top_words = [feature_names[j] for j in top_indices]
            top_weights = original_space_centroid[top_indices]
            
            for word, weight in zip(top_words, top_weights):
                cluster_data.append({"Cluster": f"Cluster {i}", "Word": word, "Weight": weight})
        
        df_clusters = pd.DataFrame(cluster_data)
        
        # Plotly Bar Chart
        fig = px.bar(
            df_clusters, 
            x="Weight", 
            y="Word", 
            color="Cluster", 
            orientation='h',
            facet_col="Cluster", 
            facet_col_wrap=2,
            height=300 * (n_clusters // 2 + 1),
            title="Top Keywords per Cluster"
        )
        # Update layout to make it look nicer
        fig.update_yaxes(matches=None, showticklabels=True)
        fig.update_xaxes(matches=None)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        
        st.plotly_chart(fig)

    # --- TAB 3: DATASET ---
    with tab_dataset:
        st.subheader("Dataset Info")
        st.write(f"**Total Documents:** {len(data.data)}")
        st.write(f"**Categories:** {len(data.target_names)}")
        with st.expander("Show Categories"):
            st.write(data.target_names)
        
        st.write("### Sample Documents")
        if st.button("Show Random Sample"):
            idx = np.random.randint(0, len(data.data))
            st.info(f"Document ID: {idx}\n\nCategory: {data.target_names[data.target[idx]]}")
            st.text(data.data[idx])

if __name__ == "__main__":
    main()
