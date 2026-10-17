"""Dataset-level Hungarian matching and masked TAS metrics.

Adapted from LTContext and MS-TCN evaluation. Opposite-partition labels are
ignored independently in predictions and targets without deleting frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import Protocol
from .discovery import intervals

IGNORED_LABEL = np.iinfo(np.int64).min


def edit_score(
    prediction: np.ndarray, target: np.ndarray, ignored_classes: tuple[int, ...] = ()
) -> float:
    predicted = [
        prediction[start]
        for start, _ in intervals(prediction)
        if prediction[start] not in ignored_classes
    ]
    truth = [
        target[start]
        for start, _ in intervals(target)
        if target[start] not in ignored_classes
    ]
    row = list(range(len(truth) + 1))
    for i, label in enumerate(predicted, start=1):
        updated = [i]
        for j, expected in enumerate(truth, start=1):
            updated.append(
                min(updated[-1] + 1, row[j] + 1, row[j - 1] + (label != expected))
            )
        row = updated
    longest = max(len(predicted), len(truth))
    return 100 * (1 - row[-1] / longest) if longest else 100.0


def segment_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    overlap: float,
    ignored_classes: tuple[int, ...] = (),
) -> tuple[int, int, int]:
    truth = [
        (start, end)
        for start, end in intervals(target)
        if target[start] not in ignored_classes
    ]
    hits = set()
    tp = fp = 0
    for start, end in intervals(prediction):
        if prediction[start] in ignored_classes:
            continue
        scores = [
            (
                max(0, min(end, stop) - max(start, begin))
                / (max(end, stop) - min(start, begin))
                if prediction[start] == target[begin]
                else 0.0
            )
            for begin, stop in truth
        ]
        if scores:
            best = int(np.argmax(scores))
            if scores[best] >= overlap and best not in hits:
                tp += 1
                hits.add(best)
                continue
        fp += 1
    return tp, fp, len(truth) - len(hits)


def evaluate(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    protocol: Protocol,
    ignore_labels: list[str],
    *,
    skip_empty_targets: bool = False,
) -> dict:
    """Score one fold after a single dataset-wide unknown-cluster assignment.

    Breakfast evaluation skips partition-empty target videos; FS
    retains them, including their false positives. The caller selects this
    explicitly rather than inferring it from a metadata directory name.
    """
    if predictions.keys() != targets.keys():
        raise ValueError("prediction and annotation video sets differ")
    known = np.asarray(protocol.known_ids)
    ignored = [
        index for index, name in protocol.labels.items() if name in ignore_labels
    ]
    cluster_ids, unknown_ids = set(), set()
    for name, truth in targets.items():
        prediction = predictions[name]
        if prediction.shape != truth.shape:
            raise ValueError(f"{name}: prediction and annotation lengths differ")
        cluster_ids.update(
            prediction[(prediction == 0) | (prediction > len(known))].tolist()
        )
        unknown_ids.update(truth[~np.isin(truth, np.r_[known, ignored])].tolist())
    clusters, classes = sorted(cluster_ids), sorted(unknown_ids)
    confusion = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    for prediction, truth in (
        (predictions[name], truth) for name, truth in targets.items()
    ):
        for i, cluster in enumerate(clusters):
            selected = truth[prediction == cluster]
            for j, action in enumerate(classes):
                confusion[i, j] += np.count_nonzero(selected == action)
    assignment = {}
    if confusion.size:
        rows, columns = linear_sum_assignment(-confusion)
        assignment = {clusters[i]: classes[j] for i, j in zip(rows, columns)}
    aligned = {}
    for name, prediction in predictions.items():
        # Keep unmatched clusters distinct: merging them would erase false-positive
        # segment boundaries and inflate F1 when K exceeds the annotated count.
        result = -prediction - 1
        for internal, original in enumerate(known, start=1):
            result[prediction == internal] = original
        for cluster, action in assignment.items():
            result[prediction == cluster] = action
        aligned[name] = result
    metrics = {}
    for partition in ["known", "unknown"]:
        correct = frames = 0
        edits, counts = [], np.zeros((3, 3), dtype=np.int64)
        for name, truth in targets.items():
            expected, predicted = truth.copy(), aligned[name].copy()
            for values in (expected, predicted):
                selected = np.isin(values, known)
                if partition == "unknown":
                    selected = ~selected
                selected &= ~np.isin(values, ignored)
                values[~selected] = IGNORED_LABEL
            mask = expected != IGNORED_LABEL
            if skip_empty_targets and not mask.any():
                continue
            correct += int(np.count_nonzero(expected[mask] == predicted[mask]))
            frames += int(mask.sum())
            edits.append(edit_score(predicted, expected, (IGNORED_LABEL,)))
            for index, threshold in enumerate([0.1, 0.25, 0.5]):
                counts[index] += segment_counts(
                    predicted, expected, threshold, (IGNORED_LABEL,)
                )
        scores = {
            "MoF": 100 * correct / frames if frames else 0.0,
            "Edit": float(np.mean(edits)) if edits else 0.0,
            "frames": frames,
        }
        for threshold, (tp, fp, fn) in zip([10, 25, 50], counts):
            scores[f"F1@{threshold}"] = (
                100 * 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            )
        metrics[partition] = scores
    return {
        "skip_empty_targets": skip_empty_targets,
        "metrics": metrics,
        "unknown_assignment": {str(k): v for k, v in assignment.items()},
    }


def summarize_folds(results: list[dict]) -> dict:
    """Arithmetic mean of per-fold percentages.

    Call once per protocol, with exactly one result per fold. Frame counts and
    assignments are deliberately not averaged or pooled across folds.
    """
    if not results:
        raise ValueError("provide at least one fold result")
    required = ("MoF", "Edit", "F1@10", "F1@25", "F1@50")
    if any(
        "skip_empty_targets" not in result
        or any(
            metric not in result.get("metrics", {}).get(partition, {})
            for partition in ("known", "unknown")
            for metric in required
        )
        for result in results
    ):
        raise ValueError(
            "fold results must contain both partitions and all reported metrics"
        )
    if len({result["skip_empty_targets"] for result in results}) != 1:
        raise ValueError("fold results use different empty-target policies")
    metrics = {}
    for partition in ("known", "unknown"):
        metrics[partition] = {}
        for metric in required:
            values = [result["metrics"][partition][metric] for result in results]
            if not np.isfinite(values).all():
                raise ValueError("fold metrics must be finite")
            metrics[partition][metric] = float(np.mean(values))
    return {
        "skip_empty_targets": results[0]["skip_empty_targets"],
        "folds": len(results),
        "metrics": metrics,
    }
