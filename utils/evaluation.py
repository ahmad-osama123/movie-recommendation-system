"""
Evaluation Module
Computes RMSE, MAE, Precision@K, Recall@K, F1@K for all models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as sk_split


def rating_metrics(predictions: list) -> dict:
    """
    predictions: list of (true_rating, predicted_rating) tuples
    Returns RMSE and MAE.
    """
    y_true = np.array([p[0] for p in predictions])
    y_pred = np.array([p[1] for p in predictions])
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


def precision_recall_f1_at_k(
    recommended_ids: list, relevant_ids: set, k: int = 10
) -> dict:
    """
    Precision@K, Recall@K, F1@K for a single user.

    recommended_ids : ordered list of recommended movieIds
    relevant_ids    : set of movieIds the user actually rated highly (≥ threshold)
    """
    top_k = recommended_ids[:k]
    hits = len(set(top_k) & relevant_ids)

    precision = hits / k if k > 0 else 0
    recall = hits / len(relevant_ids) if relevant_ids else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {
        f"Precision@{k}": round(precision, 4),
        f"Recall@{k}": round(recall, 4),
        f"F1@{k}": round(f1, 4),
    }


def evaluate_cf_model(cf_model, ratings: pd.DataFrame) -> dict:
    """Evaluate collaborative filter on its internal test set."""
    return cf_model.evaluate()


def evaluate_ranking(
    model,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    sample_users: int = 50,
    model_type: str = "hybrid",
    alpha: float = 0.7,
) -> dict:
    """
    Evaluate Precision@K, Recall@K, F1@K averaged over sampled users.

    model_type: 'hybrid', 'cf', or 'cb'
    """
    # Split: use 80% for training, 20% as ground truth
    train_ratings, test_ratings = sk_split(ratings, test_size=0.2, random_state=42)

    users = test_ratings["userId"].unique()
    if len(users) > sample_users:
        users = np.random.default_rng(42).choice(users, sample_users, replace=False)

    all_p, all_r, all_f1 = [], [], []

    for uid in users:
        # Ground truth: movies user rated highly in test set
        user_test = test_ratings[
            (test_ratings["userId"] == uid) & (test_ratings["rating"] >= relevance_threshold)
        ]
        if user_test.empty:
            continue
        relevant = set(user_test["movieId"])

        try:
            if model_type == "hybrid":
                recs = model.recommend(user_id=uid, ratings_df=train_ratings, top_n=k)
                rec_ids = recs["movieId"].tolist()
            elif model_type == "cf":
                rated = set(train_ratings[train_ratings["userId"] == uid]["movieId"])
                recs = model.get_recommendations(user_id=uid, rated_movie_ids=rated, top_n=k)
                rec_ids = recs["movieId"].tolist()
            elif model_type == "cb":
                user_train_ratings = train_ratings[train_ratings["userId"] == uid]
                recs = model.get_recommendations_for_user(
                    user_ratings=user_train_ratings, movies=movies, top_n=k
                )
                rec_ids = recs["movieId"].tolist()
            else:
                continue
        except Exception:
            continue

        metrics = precision_recall_f1_at_k(rec_ids, relevant, k=k)
        all_p.append(metrics[f"Precision@{k}"])
        all_r.append(metrics[f"Recall@{k}"])
        all_f1.append(metrics[f"F1@{k}"])

    return {
        f"Precision@{k}": round(np.mean(all_p), 4) if all_p else 0,
        f"Recall@{k}": round(np.mean(all_r), 4) if all_r else 0,
        f"F1@{k}": round(np.mean(all_f1), 4) if all_f1 else 0,
        "users_evaluated": len(all_p),
    }


def full_evaluation_report(cf, cb, hybrid, ratings, movies, k=10) -> pd.DataFrame:
    """
    Run complete evaluation for all three models and return a summary DataFrame.
    """
    print("Evaluating CF (RMSE/MAE)...")
    cf_rating = evaluate_cf_model(cf, ratings)

    print(f"Evaluating CF ranking @{k}...")
    cf_rank = evaluate_ranking(cf, ratings, movies, k=k, model_type="cf")

    print(f"Evaluating CB ranking @{k}...")
    cb_rank = evaluate_ranking(cb, ratings, movies, k=k, model_type="cb")

    print(f"Evaluating Hybrid ranking @{k}...")
    hybrid_rank = evaluate_ranking(hybrid, ratings, movies, k=k, model_type="hybrid")

    rows = [
        {
            "Model": "Collaborative Filtering (SVD)",
            "RMSE": cf_rating["RMSE"],
            "MAE": cf_rating["MAE"],
            f"Precision@{k}": cf_rank[f"Precision@{k}"],
            f"Recall@{k}": cf_rank[f"Recall@{k}"],
            f"F1@{k}": cf_rank[f"F1@{k}"],
        },
        {
            "Model": "Content-Based Filtering",
            "RMSE": "N/A",
            "MAE": "N/A",
            f"Precision@{k}": cb_rank[f"Precision@{k}"],
            f"Recall@{k}": cb_rank[f"Recall@{k}"],
            f"F1@{k}": cb_rank[f"F1@{k}"],
        },
        {
            "Model": "Hybrid (α=0.7)",
            "RMSE": "N/A",
            "MAE": "N/A",
            f"Precision@{k}": hybrid_rank[f"Precision@{k}"],
            f"Recall@{k}": hybrid_rank[f"Recall@{k}"],
            f"F1@{k}": hybrid_rank[f"F1@{k}"],
        },
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_processing import load_data, preprocess_movies, preprocess_ratings
    from models.content_based import ContentBasedFilter
    from models.collaborative_filter import CollaborativeFilter
    from models.hybrid_recommender import HybridRecommender

    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    cb = ContentBasedFilter().fit(movies)
    cf = CollaborativeFilter()
    cf.prepare_data(ratings)
    cf.fit()
    hybrid = HybridRecommender(cf, cb, movies, alpha=0.7)

    report = full_evaluation_report(cf, cb, hybrid, ratings, movies, k=10)
    print("\n=== Full Evaluation Report ===")
    print(report.to_string(index=False))
