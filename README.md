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

Evaluated on an 80/20 train-test split (80,000 training ratings, 20,000 held-out). Relevance threshold: rating ≥ 4. RMSE measures rating prediction accuracy; Precision@K and NDCG@K measure ranking quality on the top-K list.

| Model | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | RMSE | MAE |
|---|---|---|---|---|---|---|
| Item-Item CF | 0.0361 | 0.0342 | 0.0367 | 0.0357 | — | — |
| SVD (MF) | 0.0228 | 0.0278 | 0.0253 | 0.0321 | **0.773** | **0.612** |
| NMF (MF) | 0.0228 | 0.0278 | 0.0253 | 0.0321 | 1.819 | 1.453 |

**Key takeaways:**
- **Item-Item CF wins on ranking** (Precision@10 +45% over SVD) — finding similar items produces more immediately relevant top-K lists on this dataset
- **SVD wins on rating prediction** (RMSE 0.773 vs NMF 1.819) — latent factor regularisation generalises much better than NMF's non-negativity constraint on sparse data
- **NMF matches SVD on ranking** but badly overfits on rating prediction — its RMSE is 2.4× worse, showing the non-negativity constraint hurts generalisation here
- Low absolute Precision values are expected: with 93.7% sparsity and only ~106 ratings per user, the test set is thin and any unseen movie is a cold prediction

**Why NDCG over just Precision@K?** Precision@K treats rank 1 and rank K equally. NDCG discounts items lower in the list — it's a ranking metric, not a classification metric, and better reflects how users actually experience a recommendation list.

---

## Full-stack web application

The models are deployed as a live web app with a React frontend and FastAPI backend.

### Architecture

```
Browser (React + Vite)
    │  JWT auth  │  click events  │  star ratings
    ▼
FastAPI  ──── TTL Cache (5 min) ──── in-memory
    │
    ├── Item-Item CF model  (Variant A)
    ├── SVD model           (Variant B)
    └── SQLite  (users, ratings, impressions, clicks)
```

### API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| POST | `/auth/register` | Create account, receive JWT, get A/B variant assigned |
| POST | `/auth/login` | Login, receive JWT |
| GET | `/recommendations` | Top-N personalised recs (cached 5 min) |
| POST | `/ratings` | Rate a movie 1–5 stars (busts cache) |
| POST | `/events/click` | Record a recommendation click |
| GET | `/ab_test/metrics` | Live CTR per variant |

### A/B test design

Two models run in parallel. Users are deterministically assigned to a variant (even user ID → A, odd → B) so assignment is stable across sessions. The app tracks impressions and clicks per variant and exposes live CTR metrics.

| Variant | Model | Assignment |
|---------|-------|------------|
| A (control) | Item-Item CF | Even user IDs |
| B (treatment) | SVD Matrix Factorization | Odd user IDs |

### Caching

Recommendations are cached in memory with a 5-minute TTL, keyed on `recs:{user_id}:{variant}:{n}`. The cache is invalidated immediately when a user submits a new rating, ensuring the next request reflects their updated preferences.

---

## Project structure

```
movie-recommender/
├── backend/
│   ├── main.py               # FastAPI app, all 10 routes, lifespan startup
│   ├── db.py                 # SQLAlchemy models (User, UserRating, ClickEvent, Impression)
│   ├── auth.py               # JWT creation/validation, bcrypt password hashing
│   ├── cache.py              # TTL cache with prefix-based invalidation
│   ├── recommender.py        # Model loading, ML user mapping, cold-start fallback
│   └── ab_test.py            # Variant assignment, CTR metric aggregation
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Auth state, localStorage persistence
│   │   ├── api.js            # Typed fetch client for all API calls
│   │   └── components/
│   │       ├── Login.jsx     # Sign-in / register form
│   │       ├── MovieGrid.jsx # Recommendation grid + A/B metrics panel
│   │       └── MovieCard.jsx # Card with score bar, star rating, click tracking
│   ├── index.html
│   └── vite.config.js        # Dev proxy → FastAPI
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
├── requirements.txt             # ML dependencies
└── requirements-backend.txt     # API server dependencies
```

---

## Quickstart

### ML notebooks only

```bash
pip install -r requirements.txt
python src/data_loader.py                    # download data + build matrices
python notebooks/01_data_setup.py
python notebooks/02_collaborative_filtering.py
python notebooks/03_matrix_factorization.py
python notebooks/04_evaluation.py           # produces results table
python -m pytest tests/ -v
```

### Full-stack app

```bash
# Terminal 1 — backend (models train on startup, ~20s)
pip install -r requirements-backend.txt
uvicorn backend.main:app --reload
# → http://localhost:8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev
# → http://localhost:5173
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
