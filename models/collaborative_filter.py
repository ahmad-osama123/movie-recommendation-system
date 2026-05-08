"""
Collaborative Filtering Module
Uses SVD (Matrix Factorization) via scipy — compatible with NumPy 2.x.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import pickle


class CollaborativeFilter:
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_ids = None
        self.movie_ids = None
        self.user_index = None
        self.movie_index = None
        self.predicted_ratings = None
        self.global_mean = None
        self.train_df = None
        self.test_df = None
        self.all_movie_ids = None

    def prepare_data(self, ratings: pd.DataFrame, test_size: float = 0.2):
        self.train_df, self.test_df = train_test_split(
            ratings, test_size=test_size, random_state=self.random_state
        )
        self.all_movie_ids = ratings["movieId"].unique().tolist()
        self.user_ids = sorted(ratings["userId"].unique())
        self.movie_ids = sorted(ratings["movieId"].unique())
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_index = {mid: j for j, mid in enumerate(self.movie_ids)}
        return self

    def fit(self):
        n_users = len(self.user_ids)
        n_movies = len(self.movie_ids)
        print(f"[CollabFilter] Building {n_users}x{n_movies} user-item matrix...")
        self.global_mean = self.train_df["rating"].mean()

        rows = self.train_df["userId"].map(self.user_index)
        cols = self.train_df["movieId"].map(self.movie_index)
        data = self.train_df["rating"].values
        sparse_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_movies))

        user_means = np.zeros(n_users)
        for uid, i in self.user_index.items():
            ur = self.train_df[self.train_df["userId"] == uid]["rating"]
            user_means[i] = ur.mean() if len(ur) > 0 else self.global_mean

        dense = sparse_matrix.toarray().astype(float)
        mask = dense != 0
        for i in range(n_users):
            dense[i, mask[i]] -= user_means[i]

        k = min(self.n_factors, n_users - 1, n_movies - 1)
        print(f"[CollabFilter] Running SVD with k={k} factors...")
        U, sigma, Vt = svds(dense, k=k)
        reconstructed = np.dot(np.dot(U, np.diag(sigma)), Vt)
        for i in range(n_users):
            reconstructed[i] += user_means[i]
        self.predicted_ratings = np.clip(reconstructed, 0.5, 5.0)
        print("[CollabFilter] Training complete.")
        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_index or movie_id not in self.movie_index:
            return self.global_mean
        return round(float(self.predicted_ratings[self.user_index[user_id], self.movie_index[movie_id]]), 3)

    def get_recommendations(self, user_id: int, rated_movie_ids: set, top_n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_index:
            return pd.DataFrame(columns=["movieId", "cf_score"])
        i = self.user_index[user_id]
        preds = []
        for mid in self.all_movie_ids:
            if mid in rated_movie_ids:
                continue
            j = self.movie_index.get(mid)
            score = float(self.predicted_ratings[i, j]) if j is not None else self.global_mean
            preds.append((mid, score))
        preds.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(preds[:top_n], columns=["movieId", "cf_score"])

    def evaluate(self) -> dict:
        y_true, y_pred = [], []
        for _, row in self.test_df.iterrows():
            pred = self.predict_rating(int(row["userId"]), int(row["movieId"]))
            y_true.append(row["rating"])
            y_pred.append(pred)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return {
            "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
            "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CollaborativeFilter":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from utils.data_processing import load_data, preprocess_movies, preprocess_ratings

    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    cf = CollaborativeFilter(n_factors=50)
    cf.prepare_data(ratings, test_size=0.2)
    cf.fit()

    metrics = cf.evaluate()
    print(f"\n=== Collaborative Filtering Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    rated = set(ratings[ratings["userId"] == 1]["movieId"])
    recs = cf.get_recommendations(user_id=1, rated_movie_ids=rated, top_n=10)
    recs = recs.merge(movies[["movieId", "title", "genres"]], on="movieId")
    print("\n=== CF Recommendations for User 1 ===")
    print(recs[["title", "genres", "cf_score"]].to_string(index=False))
