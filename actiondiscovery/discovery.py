"""GGSM interval proposals and training-only UASA fitting, independent of labels at inference."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from .config import DiscoveryConfig


def intervals(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous intervals with exclusive end indices, including singletons."""
    if not len(labels):
        return []
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return list(zip(cuts[:-1].tolist(), cuts[1:].tolist()))


def proposal_score(
    proposals: list[tuple[int, int]], known: list[tuple[int, int]], alpha: float
) -> float:
    """Paper Eq. (4)-(6): mean pairwise IoU with an exponential length penalty."""
    if not proposals or not known:
        return 0.0
    first, second = np.asarray(proposals), np.asarray(known)
    overlap = np.maximum(
        0,
        np.minimum(first[:, None, 1], second[None, :, 1])
        - np.maximum(first[:, None, 0], second[None, :, 0]),
    )
    union = np.maximum(first[:, None, 1], second[None, :, 1]) - np.minimum(
        first[:, None, 0], second[None, :, 0]
    )
    difference = np.abs(
        (first[:, 1] - first[:, 0])[:, None] - (second[:, 1] - second[:, 0])[None]
    )
    return float(np.mean(overlap / union * np.exp(-alpha * difference)))


def ggsm(
    embeddings: np.ndarray, reference: np.ndarray, config: DiscoveryConfig
) -> list[tuple[int, int]]:
    """Choose complete-linkage proposals using known annotations or predictions.

    `reference` uses 0 for unknown. During fitting it contains partial training
    labels; during inference it contains backbone predictions exclusively.
    Without known intervals there is no granularity prior: keep the backbone spans.
    """
    if embeddings.ndim != 2 or len(embeddings) != len(reference):
        raise ValueError("GGSM expects [T,D] embeddings and [T] reference labels")
    if not np.isfinite(embeddings).all():
        raise ValueError("GGSM embeddings must be finite")
    known = [
        (start, end) for start, end in intervals(reference) if reference[start] > 0
    ]
    if len(embeddings) < 2 or not known:
        return intervals(reference)
    tree = linkage(embeddings, method="complete", metric="euclidean")
    distances = tree[:, 2]
    gaps = np.argsort(distances, kind="stable")[-config.hierarchy_levels :][::-1]
    candidates = [
        intervals(fcluster(tree, float(distances[index]), criterion="distance"))
        for index in gaps
    ]
    return max(
        candidates, key=lambda proposals: proposal_score(proposals, known, config.alpha)
    )


def majority(labels: np.ndarray) -> int:
    # First occurrence resolves ties
    unique, first, counts = np.unique(labels, return_index=True, return_counts=True)
    tied = np.flatnonzero(counts == counts.max())
    return int(unique[tied[np.argmin(first[tied])]])


def unknown_segments(
    embeddings: np.ndarray, reference: np.ndarray, proposals: list[tuple[int, int]]
) -> list[np.ndarray]:
    return [
        embeddings[start:end].mean(0)
        for start, end in proposals
        if majority(reference[start:end]) == 0
    ]


@dataclass
class DiscoveryState:
    centroids: np.ndarray
    bic: dict[int, float]

    @classmethod
    def fit(
        cls,
        segments: list[np.ndarray],
        dimension: int,
        config: DiscoveryConfig,
        seed: int,
    ) -> "DiscoveryState":
        if not segments:
            return cls(np.empty((0, dimension), dtype=np.float32), {})
        features = np.asarray(segments, dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[1] != dimension
            or not np.isfinite(features).all()
        ):
            raise ValueError(
                "UASA requires finite segment means with consistent dimensions"
            )
        if len(features) == 1:
            return cls(features.astype(np.float32), {})
        distinct = len(np.unique(features, axis=0))
        scores = {}
        for components in range(
            1, min(config.max_components, len(features), distinct) + 1
        ):
            mixture = GaussianMixture(
                n_components=components,
                random_state=seed,
                reg_covar=config.gmm_regularization,
            )
            try:
                mixture.fit(features)
                score = float(mixture.bic(features))
            except ValueError:
                continue
            if np.isfinite(score):
                scores[components] = score
        if not scores:
            raise ValueError("no finite GMM/BIC candidate; inspect training embeddings")
        components = min(scores, key=scores.get)
        centroids = (
            KMeans(n_clusters=components, random_state=seed, n_init=10)
            .fit(features)
            .cluster_centers_
        )
        return cls(centroids.astype(np.float32), scores)

    def predict(
        self,
        embeddings: np.ndarray,
        reference: np.ndarray,
        config: DiscoveryConfig,
        known_classes: int,
    ) -> np.ndarray:
        predictions = reference.copy()
        for start, end in ggsm(embeddings, reference, config):
            label = majority(reference[start:end])
            if label == 0 and len(self.centroids):
                mean = embeddings[start:end].mean(0)
                cluster = int(np.argmin(np.sum((self.centroids - mean) ** 2, axis=1)))
                label = known_classes + 1 + cluster
            predictions[start:end] = label
        return predictions
