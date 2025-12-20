# 📝 LSA with Randomized SVD (rSVD)

Latent Semantic Analysis (LSA) applied to the **20 Newsgroups** dataset, utilizing a custom **Randomized SVD** implementation for topic recovery and document clustering.

## 🚀 Project Overview
This project demonstrates how to extract latent concepts from a large text corpus by reducing the dimensionality of the Term-Document matrix. By using **rSVD** (Randomized Singular Value Decomposition), the system achieves significantly faster performance than classical SVD on large matrices while maintaining high approximation accuracy.

### Key Features:
* **Dataset:** 20 Newsgroups (~18,000 Usenet posts).
* **Preprocessing:** Advanced tokenization including **Snowball Stemming** and custom stop-word filtering.
* **Algorithm:** Custom rSVD implementation featuring **Power Iterations** to enhance singular value approximation.
* **Analysis:** Automated search for the optimal number of clusters using the **Silhouette Score**.
* **Visualization:** Generation of horizontal bar plots showing the most relevant keywords for each identified cluster.

## 📂 Repository Structure
* `test.py`: The main script handling the end-to-end pipeline: data loading, rSVD, clustering, and visualization.
* `stemming.py`: Logic for text preprocessing and NLTK-based tokenization.
* `utils.py`: Helper functions for clustering metrics, specifically the Silhouette Score analysis.
* `constants.py`: Centralized management of the global random seed for reproducibility.

## 🛠️ Requirements
The project requires the following Python libraries:
* `numpy`
* `matplotlib`
* `scikit-learn`
* `nltk`

## ⚙️ Technical Details

### Preprocessing & Vectorization
To focus on semantic content, the pipeline:
1.  Removes **headers, footers, and quotes** from the original posts.
2.  Applies **Snowball Stemming** to both the documents and the stop-words list.
3.  Uses `TfidfVectorizer` limited to the top 10,000 features.

### The rSVD Algorithm
The implementation in `test.py` follows a probabilistic approach for matrix decomposition with the following default parameters:
* `k = 50`: Target dimensions.
* `p = 20`: Oversampling parameter.
* `q = 10`: Power iterations to handle noise.

### Clustering Workflow
After projecting documents into the reduced space ($U_r$):
* Vectors are normalized to unit length.
* An automated scan identifies the **Best K** (between 5 and 10) based on the Silhouette Score.
* Final labels are assigned using **K-Means** clustering.

## 📊 Expected Output
Running `test.py` will generate:
1.  **Sparsity Logs:** Details about the TF-IDF matrix density.
2.  **Performance Benchmark:** Time taken by the rSVD for decomposition.
3.  **Silhouette Plot:** A visual report to justify the chosen number of clusters.
4.  **Cluster Barplots:** A grid of charts displaying the top 6 representative words for each discovered topic