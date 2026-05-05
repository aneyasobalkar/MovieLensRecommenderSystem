# Movie Recommender System

End-to-end recommendation system built on [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/). Implements item-item collaborative filtering and matrix factorization (SVD + NMF), evaluated with ranking metrics Precision@K and NDCG@K.

---

## Methods

### Item-Item Collaborative Filtering
Computes cosine similarity between item rating vectors. For a given user, scores unrated items by aggregating weighted ratings from their K most similar neighbors among already-rated items. More stable than user-user CF on this dataset — items are denser than users.

### Matrix Factorization (SVD + NMF)
Decomposes the rating matrix **R ≈ P × Qᵀ** where P (users × k) and Q (items × k) are latent factor matrices. The k dimensions capture abstract concepts like genre, mood, or era. SVD uses gradient descent with L2 regularization. NMF adds a non-negativity constraint, making factors more interpretable — and is structurally identical to the cNMF approach used in single-cell transcriptomics.

### Dataset stats
| Metric | Value |
|---|---|
| Users | 943 |
| Items | 1,682 |
| Ratings | 100,000 |
| Sparsity | ~93.7% |
| Rating scale | 1–5 |

---

## Results

Evaluated on 20% held-out test set (80/20 split). "Relevant" = rating ≥ 4. RMSE measures rating prediction accuracy; Precision@K and NDCG@K measure ranking quality.

| Model | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | RMSE |
|---|---|---|---|---|---|
| Item-Item CF | — | — | — | — | — |
| SVD (MF) | — | — | — | — | — |
| NMF (MF) | — | — | — | — | — |

> Run `python notebooks/04_evaluation.py` to populate this table with actual results.

**Why NDCG over just Precision@K?** Precision@K treats rank 1 and rank K equally. NDCG discounts items lower in the list — it's a ranking metric, not a classification metric, and better reflects how users experience a recommendation list.

---

## Project structure

```
movie-recommender/
├── src/
│   ├── data_loader.py          # download, parse, build matrices
│   ├── collaborative_filtering.py  # item-item CF
│   ├── matrix_factorization.py     # SVD + NMF via scikit-surprise
│   └── evaluation.py           # Precision@K, NDCG@K, RMSE
├── notebooks/
│   ├── 01_data_setup.py        # EDA, sparsity visualization
│   ├── 02_collaborative_filtering.py  # CF training + sanity checks
│   ├── 03_matrix_factorization.py     # MF training, factor viz, HPO
│   └── 04_evaluation.py        # full eval + results table
├── tests/
│   └── test_metrics.py         # unit tests for ranking metrics
├── data/                        # created on first run (gitignored)
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data + build matrices (~30 seconds)
python src/data_loader.py

# 3. Run notebooks in order
python notebooks/01_data_setup.py
python notebooks/02_collaborative_filtering.py
python notebooks/03_matrix_factorization.py
python notebooks/04_evaluation.py   # produces the results table

# 4. Run tests
python -m pytest tests/ -v
```

---

## Key design decisions

**Item-item over user-user CF** — Item vectors are denser (each movie rated by more users than each user has rated movies), giving more stable cosine similarity estimates. The advantage compounds on larger datasets.

**k=50 latent factors** — Validated via 5-fold cross-validation sweep over [10, 20, 50, 100, 150]. Diminishing returns beyond 50 on this dataset.

**Relevance threshold = 4.0** — Ratings of 4–5 indicate genuine preference. Using 3.0 inflates Precision@K by counting ambivalent ratings as hits.

**NDCG over MAP** — Mean Average Precision requires a full relevance ranking. NDCG@K only needs binary relevance (rated ≥ threshold) and is more interpretable in interviews.

---

## Connection to matrix factorization in other domains

The NMF approach here — decomposing an observed matrix into two non-negative factor matrices — is structurally identical to cNMF used in single-cell RNA-seq analysis. The constraint differs (non-negativity vs orthogonality in SVD), but the underlying objective (minimize reconstruction error via latent factors) is the same. The k latent dimensions in a recommender correspond to gene program dimensions in the transcriptomics setting.

---

## References

- [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) — GroupLens Research
- [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) — Koren et al. (2009)
- [scikit-surprise](https://surpriselib.com/) documentation
