# Model Evaluation Results

Time-based test split; K = 10; relevant = test rating >= 4. Ranking metrics averaged over **98 warm users** (users with training history and at least one relevant test item). 192 cold-start users are served by the popularity fallback (column *Cold NDCG*).

| Model | Precision@10 | Recall@10 | NDCG@10 | HitRate@10 | Coverage | Diversity | Novelty | RMSE | MAE | Cold NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity (baseline) | 0.1082 | 0.0830 | 0.1341 | 0.4592 | 0.0452 | 0.8258 | 1.58 | 1.0292 | 0.8296 | 0.3153 |
| Matrix Factorization (SVD) | 0.0663 | 0.0476 | 0.0818 | 0.2959 | 0.1391 | 0.8064 | 3.34 | 1.0232 | 0.8201 | 0.3153 |
| Hybrid (MF + content + popularity) | 0.1235 | 0.1058 | 0.1572 | 0.4898 | 0.0898 | 0.6873 | 2.09 | n/a | n/a | 0.3153 |

*Coverage* = share of the catalogue recommended to at least one user; *Diversity* = mean pairwise genre dissimilarity (1 - Jaccard) within a top-K list; *Novelty* = mean -log2(share of users who rated the item). RMSE/MAE are computed on all test ratings for models that predict explicit ratings (popularity predicts the Bayesian item average); the hybrid model is ranking-only.
