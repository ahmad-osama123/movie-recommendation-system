"""
Hybrid Recommendation Engine
Combines Content-Based and Collaborative Filtering via weighted averaging.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    """
    Weighted hybrid of Content-Based (CB) and Collaborative Filtering (CF).

    final_score = alpha * norm(cf_score) + (1 - alpha) * norm(cb_score)

    alpha=1.0 → pure CF
    alpha=0.0 → pure CB
    alpha=0.7 → 70% CF, 30% CB  (default, CF usually more accurate)
    """

    def __init__(self, cf_model, cb_model, movies: pd.DataFrame, alpha: float = 0.7):
        self.cf = cf_model
        self.cb = cb_model
        self.movies = movies
        self.alpha = alpha
        self.scaler = MinMaxScaler()

    def recommend(self, user_id: int, ratings_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Generate hybrid recommendations for a user.

        Parameters:
        -----------
        user_id   : target user
        ratings_df: full ratings DataFrame (to find what user has seen)
        top_n     : number of recommendations to return

        Returns a DataFrame with columns:
        movieId, title, genres, cf_score, cb_score, hybrid_score, rank
        """
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        rated_ids = set(user_ratings["movieId"])

        # ── 1. Collaborative Filtering scores ─────────────────────────────
        cf_recs = self.cf.get_recommendations(
            user_id=user_id,
            rated_movie_ids=rated_ids,
            top_n=len(self.movies),  # get all, filter later
        )

        # ── 2. Content-Based scores ────────────────────────────────────────
        cb_recs = self.cb.get_recommendations_for_user(
            user_ratings=user_ratings,
            movies=self.movies,
            top_n=len(self.movies),
        )

        # ── 3. Merge on movieId ────────────────────────────────────────────
        merged = cf_recs.merge(cb_recs[["movieId", "cb_score"]], on="movieId", how="outer")
        merged = merged.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")

        # Fill NaN scores with the minimum observed score (conservative)
        merged["cf_score"] = merged["cf_score"].fillna(merged["cf_score"].min())
        merged["cb_score"] = merged["cb_score"].fillna(0.0)

        # ── 4. Normalize both scores to [0, 1] ─────────────────────────────
        scores = merged[["cf_score", "cb_score"]].values
        scores_norm = MinMaxScaler().fit_transform(scores)
        merged["cf_norm"] = scores_norm[:, 0]
        merged["cb_norm"] = scores_norm[:, 1]

        # ── 5. Weighted average ────────────────────────────────────────────
        merged["hybrid_score"] = self.alpha * merged["cf_norm"] + (1 - self.alpha) * merged["cb_norm"]

        # ── 6. Filter out already-rated movies & sort ──────────────────────
        merged = merged[~merged["movieId"].isin(rated_ids)]
        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        merged["rank"] = range(1, len(merged) + 1)
        merged = merged.reset_index(drop=True)

        return merged[["rank", "movieId", "title", "genres", "cf_score", "cb_score", "hybrid_score"]]

    def recommend_for_new_user(self, liked_genres: list, top_n: int = 10) -> pd.DataFrame:
        """
        Cold-start: recommend based on preferred genres.
        Results are filtered to only include movies containing AT LEAST ONE selected genre.
        """
        genre_set = set(liked_genres)

        mask = self.movies["genre_list"].apply(lambda gs: bool(genre_set & set(gs)))
        seed_movies = self.movies[mask].copy()

        if seed_movies.empty:
            return pd.DataFrame()

        # بدل ما نعمل fake ratings ونشيل الـ seed movies،
        # نحسب الـ similarity مباشرة من الـ genre profile
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        movies_info = self.movies[["movieId", "title", "genres", "genre_list"]].copy()

        # ابني genre profile من متوسط الـ TF-IDF vectors للأفلام المختارة
        seed_indices = [
            self.cb.movie_index[mid]
            for mid in seed_movies["movieId"].values
            if mid in self.cb.movie_index
        ]

        if not seed_indices:
            return pd.DataFrame()

        profile = self.cb.tfidf_matrix[seed_indices].mean(axis=0)
        scores = cosine_similarity(np.asarray(profile), self.cb.tfidf_matrix).flatten()

        movies_info = movies_info.copy()
        movies_info["cb_score"] = [
            scores[self.cb.movie_index[mid]] if mid in self.cb.movie_index else 0.0
            for mid in movies_info["movieId"]
        ]

        # فلتر على الـ genres المختارة بس
        movies_info = movies_info[
            movies_info["genre_list"].apply(
                lambda gs: bool(genre_set & set(gs)) if isinstance(gs, list) else False
            )
        ]

        movies_info = movies_info.sort_values("cb_score", ascending=False).head(top_n)
        movies_info = movies_info.reset_index(drop=True)
        return movies_info[["movieId", "title", "genres", "cb_score"]]


if __name__ == "__main__":
    from utils.data_processing import load_data, preprocess_movies, preprocess_ratings
    from content_based import ContentBasedFilter
    from collaborative_filter import CollaborativeFilter

    print("Loading data...")
    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    print("Fitting Content-Based model...")
    cb = ContentBasedFilter()
    cb.fit(movies)

    print("Fitting Collaborative Filtering model...")
    cf = CollaborativeFilter()
    cf.prepare_data(ratings, test_size=0.2)
    cf.fit()

    print("Building Hybrid Recommender (alpha=0.7)...")
    hybrid = HybridRecommender(cf_model=cf, cb_model=cb, movies=movies, alpha=0.7)

    print("\n=== Hybrid Recommendations for User 1 ===")
    recs = hybrid.recommend(user_id=1, ratings_df=ratings, top_n=10)
    print(recs[["rank", "title", "genres", "hybrid_score"]].to_string(index=False))

    print("\n=== Cold-Start Recommendations (Action + Sci-Fi fan) ===")
    cold = hybrid.recommend_for_new_user(liked_genres=["Action", "Sci-Fi"], top_n=10)
    print(cold[["title", "genres", "cb_score"]].to_string(index=False))
