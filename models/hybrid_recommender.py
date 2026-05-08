

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class HybridRecommender:
    

    def __init__(self, cf_model, cb_model, movies: pd.DataFrame, alpha: float = 0.7):
        self.cf = cf_model
        self.cb = cb_model
        self.movies = movies
        self.alpha = alpha
        self.scaler = MinMaxScaler()

    def recommend(self, user_id: int, ratings_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
       
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        rated_ids = set(user_ratings["movieId"])

        cf_recs = self.cf.get_recommendations(
            user_id=user_id,
            rated_movie_ids=rated_ids,
            top_n=len(self.movies),  # get all, filter later
        )

        cb_recs = self.cb.get_recommendations_for_user(
            user_ratings=user_ratings,
            movies=self.movies,
            top_n=len(self.movies),
        )

        merged = cf_recs.merge(cb_recs[["movieId", "cb_score"]], on="movieId", how="outer")
        merged = merged.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")

        merged["cf_score"] = merged["cf_score"].fillna(merged["cf_score"].min())
        merged["cb_score"] = merged["cb_score"].fillna(0.0)

        scores = merged[["cf_score", "cb_score"]].values
        scores_norm = MinMaxScaler().fit_transform(scores)
        merged["cf_norm"] = scores_norm[:, 0]
        merged["cb_norm"] = scores_norm[:, 1]

        merged["hybrid_score"] = self.alpha * merged["cf_norm"] + (1 - self.alpha) * merged["cb_norm"]

        merged = merged[~merged["movieId"].isin(rated_ids)]
        merged = merged.sort_values("hybrid_score", ascending=False).head(top_n)
        merged["rank"] = range(1, len(merged) + 1)
        merged = merged.reset_index(drop=True)

        return merged[["rank", "movieId", "title", "genres", "cf_score", "cb_score", "hybrid_score"]]

    def recommend_for_new_user(self, liked_genres: list, top_n: int = 10) -> pd.DataFrame:
       
        genre_set = set(liked_genres)

        mask = self.movies["genre_list"].apply(lambda gs: bool(genre_set & set(gs)))
        seed_movies = self.movies[mask].copy()

        if seed_movies.empty:
            return pd.DataFrame()

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        movies_info = self.movies[["movieId", "title", "genres", "genre_list"]].copy()

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
