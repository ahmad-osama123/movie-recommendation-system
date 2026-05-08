# Hybrid Movie Recommendation System

## Overview
A hybrid recommendation system combining **Collaborative Filtering (SVD)** and **Content-Based Filtering (TF-IDF + Cosine Similarity)** on the MovieLens 100K dataset.

## Project Structure
```
├── data_processing.py      # Data loading & preprocessing
├── content_based.py        # TF-IDF + Cosine Similarity (CBF)
├── collaborative_filter.py # SVD Matrix Factorization (CF)
├── hybrid_recommender.py   # Weighted Hybrid Engine
├── evaluation.py           # RMSE, MAE, Precision/Recall/F1 evaluation
├── app.py                  # Streamlit interactive UI
├── movies.csv              # MovieLens movie data
├── ratings.csv             # MovieLens ratings data
├── requirements.txt        # Python dependencies
└── Evaluation_Report.docx  # Full evaluation report
```

## Setup
```bash
pip install -r requirements.txt
```

## Running the Streamlit App
```bash
streamlit run app.py
```

## Running Individual Modules
```bash
python data_processing.py       # Test data loading
python content_based.py         # Test CBF
python collaborative_filter.py  # Test CF (SVD)
python hybrid_recommender.py    # Test hybrid engine
python evaluation.py            # Full evaluation report
```

## Evaluation Results

| Model | RMSE | MAE | Precision@10 | Recall@10 | F1@10 |
|-------|------|-----|-------------|-----------|-------|
| Collaborative Filtering (SVD) | 0.9273 | 0.7168 | 0.0915 | 0.0843 | 0.0680 |
| Content-Based Filtering | N/A | N/A | 0.0064 | 0.0038 | 0.0032 |
| Hybrid (α=0.7) | N/A | N/A | **0.1106** | **0.1085** | **0.0867** |

## Key Design Decisions
- **SVD via scipy** (not Surprise) for NumPy 2.x compatibility
- **Weighted average hybrid** with configurable α (default 0.7 CF / 0.3 CB)
- **Cold-start support** for new users via genre-based seeding
- **Min-max normalization** before combining CF and CB scores
