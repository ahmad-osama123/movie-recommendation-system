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
        - Builds a separate TF-IDF profile per genre
        - Scores each movie against ALL selected genres equally
        - Ranks by balanced score across all genres (not dominated by one)
        - Only returns movies containing AT LEAST ONE selected genre
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        genre_set = set(liked_genres)
        movies_info = self.movies[["movieId", "title", "genres", "genre_list"]].copy()

        # ── 1. بناء profile منفصل لكل genre ──────────────────────────────
        genre_scores = {}
        for genre in liked_genres:
            # الأفلام اللي فيها الـ genre ده بس
            genre_movies = self.movies[
                self.movies["genre_list"].apply(lambda gs: genre in gs if isinstance(gs, list) else False)
            ]
            if genre_movies.empty:
                continue

            seed_indices = [
                self.cb.movie_index[mid]
                for mid in genre_movies["movieId"].values
                if mid in self.cb.movie_index
            ]
            if not seed_indices:
                continue

            # profile = متوسط vectors الأفلام في الـ genre ده
            profile = self.cb.tfidf_matrix[seed_indices].mean(axis=0)
            scores = cosine_similarity(np.asarray(profile), self.cb.tfidf_matrix).flatten()
            genre_scores[genre] = scores

        if not genre_scores:
            return pd.DataFrame()

        # ── 2. Score كل فيلم = متوسط scores على كل الـ genres المختارة ──
        # ده بيخلي كل genre ليها نفس الوزن بدون ما تطغى genre واحدة
        all_scores = np.stack(list(genre_scores.values()), axis=0)  # (n_genres, n_movies)
        balanced_score = all_scores.mean(axis=0)  # متوسط عادل

        movies_info["cb_score"] = [
            float(balanced_score[self.cb.movie_index[mid]]) if mid in self.cb.movie_index else 0.0
            for mid in movies_info["movieId"]
        ]

        # ── 3. لكل فيلم نحسب كام genre من المختارة موجودة فيه ──────────
        movies_info["genre_match_count"] = movies_info["genre_list"].apply(
            lambda gs: len(genre_set & set(gs)) if isinstance(gs, list) else 0
        )

        # ── 4. فلتر: لازم فيه على الأقل genre واحدة من المختارة ──────────
        movies_info = movies_info[movies_info["genre_match_count"] > 0]

        # ── 5. حساب عدد الـ genres الزيادة (مش من المطلوبة) ──────────────
        movies_info["extra_genres"] = movies_info["genre_list"].apply(
            lambda gs: len(set(gs) - genre_set) if isinstance(gs, list) else 99
        )

        # ── 6. Final Score: يفضل الأفلام اللي فيها الـ genres المطلوبة بالظبط ──
        # كلما قل عدد الـ genres الزيادة كلما ارتفعت الأولوية
        movies_info["final_score"] = (
            movies_info["cb_score"] * 0.5
            + (movies_info["genre_match_count"] / len(liked_genres)) * 0.4
            + (1 / (1 + movies_info["extra_genres"])) * 0.1
        )

        # ── 7. ترتيب حسب: كمية match أولاً ثم الـ score ──────────────────
        movies_info = movies_info.sort_values(
            ["genre_match_count", "final_score"],
            ascending=[False, False]
        ).head(top_n * 2)

        movies_info = movies_info.reset_index(drop=True)
        movies_info["cb_score"] = movies_info["final_score"]
        return movies_info[["movieId", "title", "genres", "cb_score"]].head(top_n)


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
