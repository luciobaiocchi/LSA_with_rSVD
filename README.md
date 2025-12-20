# 📝 LSA with Randomized SVD (rSVD)

Latent Semantic Analysis (LSA) applied to the **20 Newsgroups** dataset, utilizing a custom **Randomized SVD** implementation for topic recovery and document clustering.

## 🚀 Project Overview
This project demonstrates how to extract latent concepts from a large text corpus by reducing the dimensionality of the Term-Document matrix. By using **rSVD** (Randomized Singular Value Decomposition), the system achieves significantly faster performance than classical SVD on large matrices while maintaining high approximation accuracy.

### Key Features:
* **Dataset:** 20 Newsgroups (~18,000 Usenet posts).
* **Preprocessing:** Advanced tokenization including **Snowball Stemming** and custom stop-word filtering.
* **Algorithm:** Custom rSVD implementation featuring **Power Iterations** to enhance singular value approximation.
* **Analysis:** Automated search for the optimal number of clusters using the **Silhouette Score**.
* **Visualization:** 
    * **CLI**: Horizontal bar plots (Matplotlib).
    * **GUI**: Interactive charts (Plotly) and Semantic Search engine.

## 📂 Repository Structure
* `cli.py`: Command Line Interface for the pipeline (data loading, rSVD, clustering, visualization).
* `gui.py`: Streamlit Web Interface for interactive exploration and search.
* `semantic_engine.py`: Core logic for the semantic search engine and model management.
* `stemming.py`: Logic for text preprocessing and NLTK-based tokenization.
* `utils.py`: Helper functions for clustering metrics (Silhouette Score) and plotting.
* `functions.py`: Implementation of the rSVD algorithm.
* `constants.py`: Global constants (random seed).

## 🛠️ Requirements
The project requires the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `nltk`
* `streamlit` (for GUI)
* `plotly` (for GUI)

Install them via pip:
```bash
pip install numpy matplotlib scikit-learn nltk streamlit plotly
```

## 💻 Usage

### 1. Graphical User Interface (GUI) - **Recommended**
The GUI provides a modern, interactive experience to search the dataset and visualize clusters dynamically.

**How to run:**
```bash
streamlit run gui.py
```
> **Note:** Do not run with `python gui.py`. Use the `streamlit run` command.

**Features:**
* **Semantic Search:** Type queries (e.g., "space", "graphics") to find relevant documents.
* **Interactive Visualization:** Explore cluster keywords with Plotly charts.
* **Dataset Preview:** Browse raw documents.

### 2. Command Line Interface (CLI)
The CLI runs the full pipeline, performs benchmarks, and generates static plots.

**How to run:**
```bash
python cli.py
```

**Workflow:**
1.  Loads/Downloads the dataset.
2.  Trains the model (TF-IDF + rSVD + K-Means).
3.  Displays clustering analysis and generated keywords.
4.  Enters an interactive loop where you can input search queries directly in the terminal.

## ⚙️ Technical Details

### Preprocessing & Vectorization
To focus on semantic content, the pipeline:
1.  Removes **headers, footers, and quotes** from the original posts.
2.  Applies **Snowball Stemming** to both the documents and the stop-words list.
3.  Uses `TfidfVectorizer` limited to the top 15,000 features.

### The rSVD Algorithm
The implementation in `functions.py` follows a probabilistic approach for matrix decomposition:
* `k = 50`: Target dimensions.
* `p = 20`: Oversampling parameter.
* `q = 2`: Power iterations to handle noise (optimized for speed).

### Clustering Workflow
After projecting documents into the reduced space ($U_r$):
* Vectors are normalized to unit length.
* An automated scan identifies the **Best K** based on the Silhouette Score.
* Final labels are assigned using **K-Means** clustering.