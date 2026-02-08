from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from indicators.momentum import roc
from indicators.statistics import zscore
from indicators.volatility import rolling_volatility
from indicators.volume import rvol
from storage.history_manager import SessionStats, DaySummary

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
except Exception:
    PCA = None
    Isomap = None


@dataclass
class RegimeResult:
    label: str
    cluster_id: int
    cluster_size: int
    total_samples: int


@dataclass
class NeighborResult:
    label: str
    distance: float


@dataclass
class ManifoldNeighbors:
    name: str
    neighbors: List[NeighborResult]
    note: Optional[str] = None


def describe_regimes(regime_count: int) -> Dict[str, str]:
    """
    Return simple, user-facing explanations for regimes and manifold outputs.
    """
    regime_count = max(1, int(regime_count))
    regimes: Dict[str, str] = {}
    for idx in range(regime_count):
        label = f"Regime {idx + 1}"
        regimes[label] = (
            "A group of past days that behaved similarly across sessions. "
            "It does not mean bullish or bearish by itself; it is a pattern bucket."
        )

    return {
        "regime_meaning": (
            "A regime is a repeating market behavior pattern. The app groups similar days "
            "together so you can see which pattern today most resembles."
        ),
        "regime_labels": " ".join([f"{k}: {v}" for k, v in regimes.items()]),
        "isomap": (
            "Isomap builds a curved map of day features so that distance means "
            "behavioral similarity. Nearest days are the closest matches to today."
        ),
        "pca": (
            "Linear autoencoder (PCA) compresses the day features into a simpler space. "
            "Nearest days are the closest matches using that compressed view."
        ),
        "spd": (
            "SPD distance compares the intraday covariance structure versus the prior day. "
            "Lower distance means the internal price/volume behavior is more similar."
        ),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _session_order() -> List[str]:
    return ["Asia", "London", "US"]


def build_session_feature_vector(sessions: Dict[str, SessionStats]) -> Optional[np.ndarray]:
    if not sessions:
        return None

    highs = [sess.high for sess in sessions.values() if sess is not None]
    lows = [sess.low for sess in sessions.values() if sess is not None]
    if not highs or not lows:
        return None

    day_high = max(highs)
    day_low = min(lows)
    day_range = max(day_high - day_low, 1e-6)

    total_volume = sum(sess.volume for sess in sessions.values() if sess is not None)

    ordered = [sessions.get(name) for name in _session_order() if sessions.get(name) is not None]
    day_open = ordered[0].open if ordered else list(sessions.values())[0].open
    day_close = ordered[-1].close if ordered else list(sessions.values())[-1].close
    day_return = _safe_div(day_close - day_open, day_range)

    features: List[float] = [float(np.log1p(day_range)), float(np.log1p(total_volume)), day_return]

    for name in _session_order():
        sess = sessions.get(name)
        if sess is None:
            features.extend([0.0, 0.0, 0.0])
            continue
        range_ratio = _safe_div(sess.range, day_range)
        direction_ratio = _safe_div(sess.close - sess.open, day_range)
        volume_ratio = _safe_div(sess.volume, total_volume)
        features.extend([range_ratio, direction_ratio, volume_ratio])

    return np.array(features, dtype=float)


def build_feature_matrix(summaries: List[DaySummary]) -> Tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    labels: List[str] = []
    for summary in summaries:
        vec = build_session_feature_vector(summary.sessions)
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(f"{summary.date} | {summary.symbol}")
    if not vectors:
        return np.empty((0, 0)), []
    return np.vstack(vectors), labels


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, np.array([]), np.array([])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def apply_standardization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    return (X - mean) / std


def kmeans_fit(X: np.ndarray, k: int, seed: int = 42, max_iter: int = 50) -> np.ndarray:
    if X.size == 0:
        return np.empty((0, 0))
    k = max(1, min(k, len(X)))
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].copy()
    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        new_centroids = np.zeros_like(centroids)
        for idx in range(k):
            members = X[labels == idx]
            if len(members) == 0:
                new_centroids[idx] = centroids[idx]
            else:
                new_centroids[idx] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def kmeans_predict(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    if X.size == 0 or centroids.size == 0:
        return np.array([], dtype=int)
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return distances.argmin(axis=1)


def cluster_labels(labels: List[str], cluster_ids: np.ndarray) -> Dict[int, List[str]]:
    """
    Group labels by cluster id for user-facing summaries.
    """
    clusters: Dict[int, List[str]] = {}
    if cluster_ids is None or len(labels) != len(cluster_ids):
        return clusters
    for label, cluster_id in zip(labels, cluster_ids):
        cluster_id_int = int(cluster_id)
        clusters.setdefault(cluster_id_int, []).append(label)
    return clusters


def isomap_embedding(X: np.ndarray, n_components: int = 2, n_neighbors: int = 5) -> Optional[np.ndarray]:
    if Isomap is None or X.size == 0 or len(X) < 2:
        return None
    n_neighbors = max(2, min(n_neighbors, len(X) - 1))
    model = Isomap(n_components=min(n_components, X.shape[1]), n_neighbors=n_neighbors)
    return model.fit_transform(X)


def pca_embedding(X: np.ndarray, n_components: int = 2) -> Optional[np.ndarray]:
    if PCA is None or X.size == 0 or len(X) < 2:
        return None
    n_components = min(n_components, X.shape[1])
    model = PCA(n_components=n_components, random_state=42)
    return model.fit_transform(X)


def nearest_neighbors(
    embedding: np.ndarray,
    labels: List[str],
    target_index: int,
    k: int = 3,
) -> List[NeighborResult]:
    if embedding is None or embedding.size == 0 or len(labels) != len(embedding):
        return []
    distances = np.linalg.norm(embedding - embedding[target_index], axis=1)
    results = []
    for idx in np.argsort(distances):
        if idx == target_index:
            continue
        results.append(NeighborResult(label=labels[idx], distance=float(distances[idx])))
        if len(results) >= k:
            break
    return results


def spd_covariance_from_intraday(df: pd.DataFrame) -> Optional[np.ndarray]:
    if df is None or df.empty:
        return None
    if "close" not in df.columns:
        return None
    returns = df["close"].pct_change()
    vol = rolling_volatility(df["close"], length=20)
    roc_series = roc(df["close"], length=10)
    z_series = zscore(df["close"], length=20)

    if "volume" in df.columns:
        rvol_series = rvol(df, length=20)
    else:
        rvol_series = pd.Series(index=df.index, data=np.nan)

    data = pd.concat([returns, vol, roc_series, z_series, rvol_series], axis=1).dropna()
    if len(data) < 5:
        return None
    cov = np.cov(data.values.T)
    if cov.shape[0] != cov.shape[1]:
        return None
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    return cov


def _spd_logm(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 1e-12, None)
    log_vals = np.log(vals)
    return vecs @ np.diag(log_vals) @ vecs.T


def spd_log_euclidean_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    log_a = _spd_logm(a)
    log_b = _spd_logm(b)
    diff = log_a - log_b
    return float(np.linalg.norm(diff, ord="fro"))
