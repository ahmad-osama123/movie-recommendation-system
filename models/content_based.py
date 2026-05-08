"""
Content-Based Filtering Module
Uses TF-IDF on genres + cosine similarity to find similar movies.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        self.tfidf_matrix = None
        self.movies = None
        self.movie_index = None  # movieId → matrix row index

    def fit(self, movies: pd.DataFrame):
        """
        Build TF-IDF matrix from movie genres.
        Each movie's genre string (pipe-separated) becomes a document.
        """
        self.movies = movies.reset_index(drop=True).copy()
        self.movie_index = {mid: idx for idx, mid in enumerate(self.movies["movieId"])}

        import re
        import numpy as np
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer

        # TF-IDF على الـ genres (وزن عالي)
        genre_tfidf = TfidfVectorizer(token_pattern=r"[^|]+")
        genre_matrix = genre_tfidf.fit_transform(self.movies["genres"])
        self.tfidf = genre_tfidf  # للاستخدام في get_recommendations_for_user

        # TF-IDF على الـ title
        def clean_title(t):
            t = re.sub(r"[(]\d{4}[)]", "", t)
            t = re.sub(r"\b(the|a|an|of|in|and|or|to|is|on|at|by)\b", "", t, flags=re.IGNORECASE)
            return " ".join(t.strip().split())

        title_tfidf = TfidfVectorizer()
        title_matrix = title_tfidf.fit_transform(self.movies["title"].apply(clean_title))

        # دمج: genres بوزن 3 والـ title بوزن 1
        self.tfidf_matrix = hstack([genre_matrix * 3, title_matrix])
        print(f"[ContentBased] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        return self

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Return top_n movies most similar to the given movie_id.
        Returns a DataFrame with movieId, title, genres, similarity_score.
        """
        if movie_id not in self.movie_index:
            raise ValueError(f"Movie ID {movie_id} not found in the dataset.")

        idx = self.movie_index[movie_id]
        movie_vec = self.tfidf_matrix[idx]

        # Compute cosine similarity with all movies
        scores = cosine_similarity(movie_vec, self.tfidf_matrix).flatten()

        # Sort descending, exclude the movie itself
        similar_indices = np.argsort(scores)[::-1]
        similar_indices = [i for i in similar_indices if i != idx][:top_n]

        result = self.movies.iloc[similar_indices][["movieId", "title", "genres"]].copy()
        result["similarity_score"] = scores[similar_indices]
        result = result.reset_index(drop=True)
        return result

    def get_recommendations_for_user(
        self, user_ratings: pd.DataFrame, movies: pd.DataFrame, top_n: int = 10
    ) -> pd.DataFrame:
        """
        Given a user's rated movies (userId, movieId, rating),
        build a weighted genre profile and recommend unseen movies.
        """
        # Filter to only movies we know
        known = user_ratings[user_ratings["movieId"].isin(self.movie_index)]

        if known.empty:
            return pd.DataFrame()

        # Build weighted TF-IDF profile
        profile = np.zeros(self.tfidf_matrix.shape[1])
        for _, row in known.iterrows():
            idx = self.movie_index[row["movieId"]]
            weight = row["rating"] - 2.5  # center around neutral
            profile += weight * self.tfidf_matrix[idx].toarray().flatten()

        # Normalize profile
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm

        # Score all movies
        scores = cosine_similarity(profile.reshape(1, -1), self.tfidf_matrix).flatten()

        # Exclude already-rated movies
        rated_ids = set(user_ratings["movieId"])
        result_df = self.movies.copy()
        result_df["cb_score"] = scores
        result_df = result_df[~result_df["movieId"].isin(rated_ids)]
        result_df = result_df.sort_values("cb_score", ascending=False).head(top_n)
        result_df = result_df.reset_index(drop=True)
        return result_df[["movieId", "title", "genres", "cb_score"]]


if __name__ == "__main__":
    from utils.data_processing import load_data, preprocess_movies, preprocess_ratings

    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    cb = ContentBasedFilter()
    cb.fit(movies)

    # Example: find movies similar to Toy Story (movieId=1)
    print("\n=== Movies similar to Toy Story (1995) ===")
    similar = cb.get_similar_movies(movie_id=1, top_n=10)
    print(similar[["title", "genres", "similarity_score"]].to_string(index=False))

    # Example: recommendations for user 1
    user1_ratings = ratings[ratings["userId"] == 1]
    print("\n=== Content-Based Recommendations for User 1 ===")
    recs = cb.get_recommendations_for_user(user1_ratings, movies, top_n=10)
    print(recs.to_string(index=False))
