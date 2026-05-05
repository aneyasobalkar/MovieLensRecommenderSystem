"""
matrix_factorization.py
Matrix factorization recommender using SVD (and optionally NMF) via
the scikit-surprise library.

Connection to cNMF research:
    Both NMF and SVD decompose a rating matrix R ≈ P × Q^T where
    P (n_users × k) and Q (n_items × k) are latent factor matrices.
    The k latent dimensions capture abstract concepts (genre, mood, era).
    NMF adds a non-negativity constraint — identical mathematical structure
    to the cNMF approach used in transcriptomics / single-cell analysis.
    The only substantive difference is the constraint and solver, not the idea.
"""

import numpy as np
import pandas as pd
from surprise import SVD, NMF, Dataset, Reader
from surprise.model_selection import train_test_split as surprise_split
from typing import Literal


class MFRecommender:
    """
    Matrix factorization recommender wrapping scikit-surprise SVD / NMF.

    Usage:
        mf = MFRecommender(algorithm="svd", n_factors=50)
        mf.fit(ratings_df)
        recs = mf.recommend(user_id=42, all_movie_ids=movie_ids, n=10)
    """

    def __init__(
        self,
        algorithm: Literal["svd", "nmf"] = "svd",
        n_factors: int = 50,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        random_state: int = 42,
    ):
        """
        Args:
            algorithm: 'svd' (default, best accuracy) or 'nmf' (non-negative,
                       directly analogous to cNMF).
            n_factors: number of latent dimensions (try 20, 50, 100).
            n_epochs: SGD training epochs.
            lr_all: learning rate for all parameters.
            reg_all: regularization for all parameters.
        """
        self.algorithm = algorithm
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.model = None
        self._trainset = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, ratings: pd.DataFrame) -> "MFRecommender":
        """
        Train on a ratings DataFrame with columns [user_id, movie_id, rating].

        Note: surprise trains on the full dataset here. For held-out evaluation
        use train_test_split() below and call fit() on the trainset directly.
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings[["user_id", "movie_id", "rating"]], reader
        )
        trainset = data.build_full_trainset()
        self._trainset = trainset

        if self.algorithm == "svd":
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=self.random_state,
                verbose=False,
            )
        else:
            self.model = NMF(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                reg_pu=self.reg_all,
                reg_qi=self.reg_all,
                random_state=self.random_state,
                verbose=False,
            )

        print(f"Training {self.algorithm.upper()} (k={self.n_factors}, epochs={self.n_epochs})...")
        self.model.fit(trainset)
        print("done.")
        return self

    def fit_trainset(self, trainset) -> "MFRecommender":
        """Fit directly on a surprise Trainset (used in cross-val / eval pipeline)."""
        self._trainset = trainset
        if self.algorithm == "svd":
            self.model = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=self.random_state,
                verbose=False,
            )
        else:
            self.model = NMF(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                reg_pu=self.reg_all,
                reg_qi=self.reg_all,
                random_state=self.random_state,
                verbose=False,
            )
        self.model.fit(trainset)
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a single (user, item) pair."""
        pred = self.model.predict(str(user_id), str(movie_id))
        return pred.est

    # ── recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        all_movie_ids: list,
        rated_movie_ids: set = None,
        n: int = 10,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-n recommendations for a user.

        Args:
            user_id: target user.
            all_movie_ids: list of all candidate movie_ids.
            rated_movie_ids: set of movie_ids user already rated (for filtering).
            n: number of recommendations.
            exclude_rated: exclude already-rated movies.

        Returns:
            DataFrame [movie_id, score] sorted descending.
        """
        scores = []
        for mid in all_movie_ids:
            if exclude_rated and rated_movie_ids and mid in rated_movie_ids:
                continue
            score = self.predict_score(user_id, mid)
            scores.append((mid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(scores[:n], columns=["movie_id", "score"])

    # ── factor inspection ─────────────────────────────────────────────────────

    def get_item_factors(self) -> np.ndarray:
        """Return the item latent factor matrix Q, shape (n_items, n_factors)."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.qi  # surprise stores item factors as .qi

    def get_user_factors(self) -> np.ndarray:
        """Return the user latent factor matrix P, shape (n_users, n_factors)."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.pu


# ── train/test split helper ───────────────────────────────────────────────────

def make_surprise_dataset(ratings: pd.DataFrame):
    """Convert ratings DataFrame to a surprise Dataset."""
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)


def split_dataset(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split ratings into surprise trainset / testset.
    Returns (trainset, testset).
    """
    data = make_surprise_dataset(ratings)
    return surprise_split(data, test_size=test_size, random_state=random_state)
