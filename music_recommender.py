"""Рекомендательные системы для музыкальных треков.

В одном файле реализованы User2User, ALS, базовые модели и оценка качества.
При запуске создается учебный набор взаимодействий, а модели сравниваются на
отложенных прослушиваниях. Для собственного датасета используйте --data.

Установка: pip install numpy pandas
Запуск: python music_recommender.py
Проверка: python music_recommender.py --self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Ranking metrics: AP@K and MAP@K from the assignment
# ---------------------------------------------------------------------------

def binary_relevance(recommended: Sequence[Hashable], relevant: Sequence[Hashable]) -> list[int]:
    """Return 0/1 labels of recommended items against the ground truth."""
    relevant_set = set(relevant)
    return [int(item in relevant_set) for item in recommended]


def ap_at_k(recommended: Sequence[Hashable], relevant: Sequence[Hashable], k: int) -> float:
    """Average Precision at K, normalized by min(K, |relevant|).

    A ranking is expected to contain unique item IDs. Repeated recommendations
    are treated as non-relevant after the first occurrence, to avoid inflated AP.
    """
    if k < 1:
        raise ValueError("k must be a positive integer")
    targets = set(relevant)
    if not targets:
        return 0.0
    hits = 0
    numerator = 0.0
    seen: set[Hashable] = set()
    for position, item in enumerate(recommended[:k], start=1):
        if item in targets and item not in seen:
            hits += 1
            numerator += hits / position
        seen.add(item)
    return numerator / min(k, len(targets))


def map_at_k(recommendations: Sequence[Sequence[Hashable]],
             true_items: Sequence[Sequence[Hashable]], k: int) -> float:
    """Mean of per-user AP@K, including users with zero relevant items as 0."""
    if len(recommendations) != len(true_items):
        raise ValueError("recommendations and true_items must have equal lengths")
    if not recommendations:
        return 0.0
    return float(np.mean([ap_at_k(pred, truth, k)
                          for pred, truth in zip(recommendations, true_items)]))


def hit_rate_at_k(recommendations: Sequence[Sequence[Hashable]],
                  true_items: Sequence[Sequence[Hashable]]) -> float:
    """Fraction of users who received at least one relevant recommendation."""
    if not recommendations:
        return 0.0
    return float(np.mean([bool(set(rec) & set(truth))
                          for rec, truth in zip(recommendations, true_items)]))


def catalog_coverage(recommendations: Sequence[Sequence[Hashable]], catalog: Sequence[Hashable]) -> float:
    """Fraction of all catalog items appearing in any top-K list."""
    if not catalog:
        return 0.0
    used = {item for rec in recommendations for item in rec}
    return len(used & set(catalog)) / len(set(catalog))


# ---------------------------------------------------------------------------
# 2. Similarity and data handling
# ---------------------------------------------------------------------------

def jaccard_sim(ratings: np.ndarray, user_vector: np.ndarray) -> np.ndarray:
    """Jaccard similarity between a binary user vector and every matrix row."""
    matrix = np.asarray(ratings, dtype=bool)
    user = np.asarray(user_vector, dtype=bool)
    if matrix.ndim != 2 or user.ndim != 1 or matrix.shape[1] != len(user):
        raise ValueError("Expected matrix (n_users, n_items) and user vector (n_items,)")
    intersection = np.logical_and(matrix, user).sum(axis=1)
    union = np.logical_or(matrix, user).sum(axis=1)
    return np.divide(intersection, union, out=np.zeros(len(intersection), dtype=float),
                     where=union != 0)


def normalize_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the input format; every row is a positive listening event."""
    renamed = df.rename(columns={'userId': 'user_id', 'trackId': 'item_id'})
    missing = {'user_id', 'item_id'} - set(renamed.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    result = renamed[['user_id', 'item_id']].dropna().drop_duplicates().reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid interactions found")
    return result


def make_synthetic_data(n_users: int = 160, n_items: int = 180,
                        n_interactions: int = 24, n_genres: int = 6,
                        seed: int = 42) -> pd.DataFrame:
    """Make illustrative user-track interactions with latent genre preferences.

    These are invented data with deliberately embedded preference structure.
    They MUST NOT be described as a real dataset or real-world performance.
    """
    if n_interactions >= n_items or n_interactions < 6 or n_genres < 2:
        raise ValueError("Require 6 <= interactions < items and at least two genres")
    rng = np.random.default_rng(seed)
    track_genres = np.arange(n_items) % n_genres
    popularity = rng.normal(0.0, 0.7, n_items)
    rows: list[tuple[int, int]] = []
    for uid in range(n_users):
        favorite = int(rng.integers(n_genres))
        secondary = (favorite + 1) % n_genres
        # A probabilistic utility with one strong and one moderate genre preference.
        utility = (2.2 * (track_genres == favorite)
                   + 0.75 * (track_genres == secondary)
                   + 0.65 * popularity + rng.normal(0.0, 0.35, n_items))
        # Gumbel-top-k is equivalent to sampling without replacement under a
        # Plackett-Luce preference model. This prevents duplicated listens.
        chosen = np.argpartition(utility + rng.gumbel(size=n_items), -n_interactions)[-n_interactions:]
        rows.extend((uid, int(item)) for item in chosen)
    return pd.DataFrame(rows, columns=['user_id', 'item_id'])


def split_by_user(df: pd.DataFrame, test_per_user: int = 4, seed: int = 42
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave out some distinct tracks per user; avoid train/test leakage."""
    if test_per_user < 1:
        raise ValueError("test_per_user must be >= 1")
    rng = np.random.default_rng(seed)
    test_indices: list[int] = []
    for user_id, part in df.groupby('user_id', sort=False):
        if len(part) <= test_per_user:
            raise ValueError(f"User {user_id!r} needs > {test_per_user} unique tracks")
        test_indices.extend(rng.choice(part.index.to_numpy(), test_per_user, replace=False))
    test = df.loc[test_indices].reset_index(drop=True)
    train = df.drop(index=test_indices).reset_index(drop=True)
    assert not (set(map(tuple, train.to_numpy())) & set(map(tuple, test.to_numpy())))
    return train, test


class InteractionMatrix:
    """Dense interaction matrix for a compact teaching/demo dataset.

    Real production datasets should use sparse matrices, not this dense form.
    """
    def __init__(self, train: pd.DataFrame, all_items: Sequence[Hashable] | None = None):
        self.users = pd.Index(pd.unique(train['user_id'])).tolist()
        self.items = list(pd.unique(train['item_id'])) if all_items is None else list(all_items)
        self.uid_to_pos = {uid: idx for idx, uid in enumerate(self.users)}
        self.item_to_pos = {item: idx for idx, item in enumerate(self.items)}
        self.R = np.zeros((len(self.users), len(self.items)), dtype=float)
        for uid, item in train[['user_id', 'item_id']].itertuples(index=False, name=None):
            self.R[self.uid_to_pos[uid], self.item_to_pos[item]] = 1.0
        self.seen = {uid: set(part['item_id']) for uid, part in train.groupby('user_id')}
        self.popularity = self.R.sum(axis=0)


# ---------------------------------------------------------------------------
# 3. Recommenders: from baselines to collaborative filtering
# ---------------------------------------------------------------------------

class BaseRecommender:
    def __init__(self, matrix: InteractionMatrix):
        self.data = matrix

    def scores(self, user_id: Hashable) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id: Hashable, k: int = 10) -> list[Hashable]:
        """Return up to k new item IDs, ordered by predicted relevance."""
        if k < 1:
            raise ValueError("k must be a positive integer")
        if user_id not in self.data.uid_to_pos:
            raise KeyError(f"Unknown user: {user_id!r}")
        prediction = self.scores(user_id).astype(float).copy()
        if prediction.shape != (len(self.data.items),):
            raise ValueError("Scores have incorrect shape")
        prediction[~np.isfinite(prediction)] = -np.inf
        for item in self.data.seen[user_id]:
            prediction[self.data.item_to_pos[item]] = -np.inf
        order = np.argsort(-prediction, kind='stable')
        selected = [self.data.items[i] for i in order if np.isfinite(prediction[i])]
        return selected[:k]


class RandomRecommender(BaseRecommender):
    def __init__(self, matrix: InteractionMatrix, seed: int = 42):
        super().__init__(matrix)
        self.seed = seed

    def scores(self, user_id: Hashable) -> np.ndarray:
        # Per-user deterministic random baseline; does not depend on query order.
        idx = self.data.uid_to_pos[user_id]
        return np.random.default_rng(self.seed + idx * 7919).random(len(self.data.items))


class PopularityRecommender(BaseRecommender):
    def scores(self, user_id: Hashable) -> np.ndarray:
        return self.data.popularity.copy()


class User2User(BaseRecommender):
    """Memory-based collaborative filtering with Jaccard and weighted votes."""
    def __init__(self, matrix: InteractionMatrix, alpha: float = 0.02):
        super().__init__(matrix)
        self.alpha = alpha

    def similarity(self, user_vector: np.ndarray) -> np.ndarray:
        return jaccard_sim(self.data.R, user_vector)

    def scores(self, user_id: Hashable) -> np.ndarray:
        idx = self.data.uid_to_pos[user_id]
        similarities = self.similarity(self.data.R[idx])
        similarities[idx] = 0.0  # a user must never be their own neighbor
        similarities[similarities <= self.alpha] = 0.0
        norm = np.abs(similarities).sum()
        if norm == 0:
            return self.data.popularity.copy()  # deterministic fallback
        return (similarities @ self.data.R) / norm


# ---------------------------------------------------------------------------
# 4. ALS steps from coursework + an explicit-feedback ALS model
# ---------------------------------------------------------------------------

def _als_user_step(items_embeddings: np.ndarray, user_ratings: np.ndarray,
                   reg_coef: float) -> np.ndarray:
    """Solve (Q'Q + lambda*I) p = Q'r for observed item ratings."""
    Q = np.asarray(items_embeddings, dtype=float)
    r = np.asarray(user_ratings, dtype=float)
    if Q.ndim != 2 or len(r) != Q.shape[0]:
        raise ValueError("Embeddings and ratings have incompatible shapes")
    if Q.shape[0] == 0:
        return np.zeros(Q.shape[1], dtype=float)
    return np.linalg.solve(Q.T @ Q + reg_coef * np.eye(Q.shape[1]), Q.T @ r)


def _als_item_step(users_embeddings: np.ndarray, item_ratings: np.ndarray,
                   reg_coef: float) -> np.ndarray:
    """Same regularized least-squares step, with users/items reversed."""
    return _als_user_step(users_embeddings, item_ratings, reg_coef)


class ObservedALS(BaseRecommender):
    """Coursework-style ALS: optimizes squared error ONLY on observed pairs.

    Important: with binary listening histories, missing interactions are not
    negative evidence. This baseline is retained to show the original method,
    not as a claim that it is the ideal objective for implicit feedback.
    """
    def __init__(self, matrix: InteractionMatrix, factors: int = 12,
                 reg: float = 0.7, epochs: int = 12, seed: int = 42):
        super().__init__(matrix)
        self.factors, self.reg, self.epochs, self.seed = factors, reg, epochs, seed
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def fit(self) -> ObservedALS:
        rng = np.random.default_rng(self.seed)
        R = self.data.R
        self.P = rng.normal(0, 0.1, size=(R.shape[0], self.factors))
        self.Q = rng.normal(0, 0.1, size=(R.shape[1], self.factors))
        for _ in range(self.epochs):
            for u in range(R.shape[0]):
                ids = np.flatnonzero(R[u])
                self.P[u] = _als_user_step(self.Q[ids], R[u, ids], self.reg)
            for i in range(R.shape[1]):
                ids = np.flatnonzero(R[:, i])
                self.Q[i] = _als_item_step(self.P[ids], R[ids, i], self.reg)
        return self

    def scores(self, user_id: Hashable) -> np.ndarray:
        if self.P is None or self.Q is None:
            raise RuntimeError("Call fit() first")
        return self.P[self.data.uid_to_pos[user_id]] @ self.Q.T


# ---------------------------------------------------------------------------
# 5. Portfolio extension: confidence-weighted ALS for implicit feedback
# ---------------------------------------------------------------------------

class ImplicitALS(BaseRecommender):
    """Alternating least squares for implicit (listened/not listened) feedback.

    For listened items: target p_ui = 1, confidence c_ui = 1 + alpha.
    For unobserved items: target p_ui = 0, confidence c_ui = 1.
    Missing items are weak negatives rather than omitted training examples.
    Reference: Hu, Koren & Volinsky (2008), Collaborative Filtering for Implicit
    Feedback Datasets. Simplified educational implementation; dense NumPy.
    """
    def __init__(self, matrix: InteractionMatrix, factors: int = 12,
                 reg: float = 0.7, alpha: float = 15.0,
                 epochs: int = 12, seed: int = 42):
        super().__init__(matrix)
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.epochs, self.seed = epochs, seed
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None

    def fit(self) -> ImplicitALS:
        rng = np.random.default_rng(self.seed)
        R = self.data.R
        n_users, n_items = R.shape
        self.P = rng.normal(0, 0.1, size=(n_users, self.factors))
        self.Q = rng.normal(0, 0.1, size=(n_items, self.factors))
        eye = np.eye(self.factors)
        for _ in range(self.epochs):
            q_gram = self.Q.T @ self.Q
            for u in range(n_users):
                idx = np.flatnonzero(R[u])
                Q_u = self.Q[idx]
                A = q_gram + self.alpha * (Q_u.T @ Q_u) + self.reg * eye
                b = (1 + self.alpha) * Q_u.sum(axis=0)
                self.P[u] = np.linalg.solve(A, b)
            p_gram = self.P.T @ self.P
            for i in range(n_items):
                idx = np.flatnonzero(R[:, i])
                P_i = self.P[idx]
                A = p_gram + self.alpha * (P_i.T @ P_i) + self.reg * eye
                b = (1 + self.alpha) * P_i.sum(axis=0)
                self.Q[i] = np.linalg.solve(A, b)
        return self

    def scores(self, user_id: Hashable) -> np.ndarray:
        if self.P is None or self.Q is None:
            raise RuntimeError("Call fit() first")
        return self.P[self.data.uid_to_pos[user_id]] @ self.Q.T


# ---------------------------------------------------------------------------
# 6. Evaluation, checks and reproducible demo
# ---------------------------------------------------------------------------

def evaluate(model: BaseRecommender, test: pd.DataFrame,
             k: int) -> tuple[dict[str, float], list[list[Hashable]]]:
    ground_truth = test.groupby('user_id')['item_id'].agg(list).to_dict()
    users = [uid for uid in model.data.users if uid in ground_truth]
    predictions = [model.recommend(uid, k) for uid in users]
    targets = [ground_truth[uid] for uid in users]
    for uid, recommended in zip(users, predictions):
        assert not (set(recommended) & model.data.seen[uid]), "Training interaction was recommended"
        assert len(recommended) == len(set(recommended)), "Repeated recommendation"
    return {
        'MAP@K': map_at_k(predictions, targets, k),
        'HitRate@K': hit_rate_at_k(predictions, targets),
        'Coverage@K': catalog_coverage(predictions, model.data.items),
    }, predictions


def run_self_tests() -> None:
    """Fast sanity checks, not a replacement for a comprehensive unit suite."""
    m = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 0]])
    assert np.allclose(jaccard_sim(m, m[0]), [1.0, 1 / 3, 0.0])
    assert abs(ap_at_k([2, 9, 4], [2, 4], 3) - 5 / 6) < 1e-12
    assert ap_at_k([1, 1, 2], [1, 2], 3) <= 1.0
    assert map_at_k([[1, 2], [3, 4]], [[1], [4]], 2) == 0.75
    assert map_at_k([], [], 10) == 0.0
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(_als_user_step(Q, np.array([2.0, 4.0]), 1.0), [1.0, 2.0])
    data = normalize_interactions(pd.DataFrame({'user_id': [10, 10, 25, 25, 50, 50],
                                                'item_id': [101, 103, 101, 105, 103, 105]}))
    matrix = InteractionMatrix(data, all_items=[101, 103, 105, 999])
    for model in (RandomRecommender(matrix), PopularityRecommender(matrix),
                  User2User(matrix, alpha=0.0), ObservedALS(matrix, factors=2, epochs=2).fit(),
                  ImplicitALS(matrix, factors=2, epochs=2).fit()):
        assert set(model.recommend(10, 5)) == {105, 999}
    print("Sanity checks: PASSED")


def run_experiment(args: argparse.Namespace) -> None:
    run_self_tests()
    synthetic = args.data is None
    if synthetic:
        interactions = make_synthetic_data(seed=args.seed)
    else:
        interactions = normalize_interactions(pd.read_csv(args.data))
    train, test = split_by_user(interactions, test_per_user=args.test_items, seed=args.seed)
    data = InteractionMatrix(train, all_items=list(pd.unique(interactions['item_id'])))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if synthetic:
        interactions.to_csv(output / 'synthetic_interactions.csv', index=False)
    train.to_csv(output / 'train.csv', index=False)
    test.to_csv(output / 'test.csv', index=False)

    models: list[tuple[str, BaseRecommender]] = [
        ('Random', RandomRecommender(data, seed=args.seed)),
        ('Popularity', PopularityRecommender(data)),
        ('User2User (Jaccard)', User2User(data, alpha=0.02)),
        ('ALS (observed-only)', ObservedALS(data, epochs=args.epochs, seed=args.seed).fit()),
        ('ALS (implicit)', ImplicitALS(data, epochs=args.epochs, seed=args.seed).fit()),
    ]

    report = []
    sample = []
    preview_users = data.users[:3]
    for name, model in models:
        metrics, _ = evaluate(model, test, args.k)
        report.append({'model': name, **metrics})
        for uid in preview_users:
            sample.append({'model': name, 'user_id': uid,
                           'top_k_unseen': ', '.join(map(str, model.recommend(uid, args.k)))})

    results = pd.DataFrame(report).sort_values('MAP@K', ascending=False).reset_index(drop=True)
    results.to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame(sample).to_csv(output / 'example_recommendations.csv', index=False)
    label = 'SYNTHETIC / ILLUSTRATIVE' if synthetic else 'USER-SUPPLIED CSV'
    title = f"MUSIC RECOMMENDATIONS - {label}\n"
    desc = (f"Users={len(data.users)}, items={len(data.items)}, "
            f"train_interactions={len(train)}, test_interactions={len(test)}, "
            f"k={args.k}, seed={args.seed}\n"
            "MAP@K, HitRate@K and Coverage@K are computed on held-out user interactions.\n"
            "These scores are not claims about real-world product performance.\n\n")
    summary = title + desc + results.to_string(index=False, float_format=lambda x: f'{x:.4f}') + '\n'
    (output / 'experiment_summary.txt').write_text(summary, encoding='utf-8')
    print('\n' + summary)
    print(f"Files saved to: {output.resolve()}")


def cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=None, type=Path,
                        help='CSV of real interactions; if omitted, generate synthetic data')
    parser.add_argument('--output-dir', default='recsys_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--test-items', type=int, default=4)
    parser.add_argument('--self-test', action='store_true', help='Run sanity checks only')
    return parser


if __name__ == '__main__':
    arguments = cli().parse_args()
    if arguments.self_test:
        run_self_tests()
    else:
        run_experiment(arguments)
