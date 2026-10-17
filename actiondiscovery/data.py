"""Feature datasets and published action partitions; raw videos are not required."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config


@dataclass
class Protocol:
    labels: dict[int, str]
    known_ids: list[int]
    directory: Path

    @classmethod
    def load(cls, directory: Path) -> "Protocol":
        labels = {}
        for line in (directory / "mapping.txt").read_text().splitlines():
            if line.strip():
                index, name = line.split(maxsplit=1)
                if int(index) < 0:
                    raise ValueError("dataset label IDs must be nonnegative")
                if int(index) in labels or name in labels.values():
                    raise ValueError(f"duplicate mapping entry: {line}")
                labels[int(index)] = name
        name_to_id = {name: index for index, name in labels.items()}
        known_names = [
            line.strip()
            for line in (directory / "set1.actions.bundle").read_text().splitlines()
            if line.strip()
        ]
        if not known_names or len(set(known_names)) != len(known_names):
            raise ValueError(
                "known-action partition must be nonempty and contain no duplicates"
            )
        try:
            known_ids = [name_to_id[name] for name in known_names]
        except KeyError as error:
            raise ValueError(f"known action absent from mapping: {error}") from error
        return cls(labels, known_ids, directory)

    def partial_targets(self, labels: np.ndarray) -> np.ndarray:
        """Map known dataset IDs to 1..N and every unannotated action to UNK=0."""
        targets = np.zeros(labels.shape, dtype=np.int64)
        for internal, original in enumerate(self.known_ids, start=1):
            targets[labels == original] = internal
        return targets

    def signature(self) -> dict:
        return {
            "labels": {str(k): v for k, v in self.labels.items()},
            "known_ids": self.known_ids,
        }


class FeatureDataset:
    """Single-video iterator. Features have shape [feature_dim, time].

    Training/discovery fitting request partial labels. Inference requests features
    alone, and scoring loads full labels only after predictions have been made.
    """

    def __init__(
        self,
        config: Config,
        protocol: Protocol,
        data_root: Path,
        split: int,
        subset: str,
        max_videos: int | None = None,
        max_frames: int | None = None,
    ):
        if not 1 <= split <= config.splits or subset not in {"train", "test"}:
            raise ValueError("invalid split or subset")
        self.config, self.protocol = config, protocol
        self.directory = data_root / config.dataset_folder
        self.max_frames = max_frames
        manifest = protocol.directory / f"{subset}.split{split}.bundle"
        self.names = [
            name.strip() for name in manifest.read_text().splitlines() if name.strip()
        ]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate video names in {manifest}")
        if any(Path(name).name != name for name in self.names):
            raise ValueError("split manifests must contain filenames, not paths")
        if max_videos:
            self.names = self.names[:max_videos]
        if not self.names:
            raise ValueError(f"no videos in {manifest}")

    def features(self, name: str) -> torch.Tensor:
        path = self.directory / "features" / f"{Path(name).stem}.npy"
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        if self.config.feature_layout == "TD":
            features = features.T
        if features.ndim != 2 or features.shape[0] != self.config.model.input_dim:
            raise ValueError(
                f"{path}: expected [D,T] with D={self.config.model.input_dim}; got {features.shape}"
            )
        features = features[:, :: self.config.sampling_rate]
        if self.max_frames:
            features = features[:, : self.max_frames]
        if features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError(f"{path}: features must be nonempty and finite")
        tensor = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
        if self.config.normalize_features:
            tensor = F.normalize(tensor, dim=0)
        return tensor

    def full_targets(self, name: str, length: int) -> np.ndarray:
        path = self.directory / "groundTruth" / name
        reverse = {name: index for index, name in self.protocol.labels.items()}
        try:
            labels = np.array(
                [
                    reverse[line.strip()]
                    for line in path.read_text().splitlines()
                    if line.strip()
                ],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(f"{path}: action absent from mapping: {error}") from error
        if not len(labels):
            raise ValueError(f"{path}: empty annotation")
        # Sample annotations with the features, then align mismatched lengths.
        labels = labels[:: self.config.sampling_rate]
        shape = np.load(
            self.directory / "features" / f"{Path(name).stem}.npy",
            mmap_mode="r",
            allow_pickle=False,
        ).shape
        original_length = shape[0 if self.config.feature_layout == "TD" else 1]
        sampled_length = (
            original_length + self.config.sampling_rate - 1
        ) // self.config.sampling_rate
        if len(labels) != sampled_length:
            labels = (
                F.interpolate(
                    torch.tensor(labels, dtype=torch.float32)[None, None],
                    size=sampled_length,
                    mode="nearest",
                )[0, 0]
                .long()
                .numpy()
            )
        return labels[:length]

    def training_sample(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(name)
        targets = self.protocol.partial_targets(
            self.full_targets(name, features.shape[1])
        )
        return features, torch.from_numpy(targets)

    def __len__(self) -> int:
        return len(self.names)
