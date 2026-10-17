"""Training, fitting discovery artifacts, and evaluating frozen predictions."""

import hashlib
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from .config import Config
from .data import FeatureDataset, Protocol
from .discovery import DiscoveryState, ggsm, unknown_segments
from .loss import training_loss
from .metrics import evaluate
from .model import MultiStageModel

LOG = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(
    config: Config,
    dataset: FeatureDataset,
    device: torch.device,
    output: Path,
    tracking=None,
) -> Path:
    model = MultiStageModel(config.model, len(dataset.protocol.known_ids)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    constant = LinearLR(optimizer, start_factor=1.0)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=config.training.cosine_epochs - config.training.warmup_epochs,
        eta_min=config.training.minimum_lr,
    )
    scheduler = (
        SequentialLR(optimizer, [constant, cosine], [config.training.warmup_epochs])
        if config.training.warmup_epochs
        else cosine
    )
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists():
        raise FileExistsError(f"{checkpoint} exists; choose a fresh output directory")
    for epoch in range(config.training.epochs):
        model.train()
        totals = {}
        names = list(dataset.names)
        random.shuffle(names)
        for name in tqdm(names, desc=f"epoch {epoch + 1}/{config.training.epochs}"):
            features, labels = dataset.training_sample(name)
            logits, embeddings = model(features[None].to(device))
            losses = training_loss(
                logits, embeddings, labels[None].to(device), config.training
            )
            if not torch.isfinite(losses["loss"]):
                raise ValueError(f"{name}: nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) / len(names)
        totals.update(epoch=epoch + 1, learning_rate=optimizer.param_groups[0]["lr"])
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(totals, allow_nan=False) + "\n")
        LOG.info("epoch %s: loss %.6f", epoch + 1, totals["loss"])
        if tracking:
            tracking.log(totals)
        scheduler.step()
        torch.save(
            {
                "format_version": 1,
                "epoch": epoch + 1,
                "config": config.to_dict(),
                "protocol": dataset.protocol.signature(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint,
        )
    return checkpoint


def load_model(
    checkpoint: Path, config: Config, protocol: Protocol, device: torch.device
) -> MultiStageModel:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if (
        payload.get("format_version") != 1
        or payload.get("protocol") != protocol.signature()
    ):
        raise ValueError("checkpoint protocol or format is incompatible.")
    stored = Config.from_dict(payload["config"])
    if (
        stored.model != config.model
        or stored.protocol != config.protocol
        or stored.sampling_rate != config.sampling_rate
        or stored.feature_layout != config.feature_layout
        or stored.normalize_features != config.normalize_features
    ):
        raise ValueError(
            "checkpoint architecture or feature preprocessing differs from the selected configuration"
        )
    model = MultiStageModel(config.model, len(protocol.known_ids)).to(device)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as error:
        raise ValueError(
            "checkpoint tensor dimensions are incompatible; weights are never silently resized"
        ) from error
    return model.eval()


@torch.no_grad()
def fit_discovery(
    model: MultiStageModel,
    dataset: FeatureDataset,
    config: Config,
    device: torch.device,
) -> DiscoveryState:
    segments = []
    for name in tqdm(dataset.names, desc="fit UASA on training videos"):
        features, reference = dataset.training_sample(name)
        _, embeddings = model(features[None].to(device))
        embeddings = embeddings[0].T.cpu().numpy()
        reference = reference.numpy()
        segments.extend(
            unknown_segments(
                embeddings, reference, ggsm(embeddings, reference, config.discovery)
            )
        )
    return DiscoveryState.fit(
        segments, config.model.hidden_dim, config.discovery, config.seed
    )


@torch.no_grad()
def predict(
    model: MultiStageModel,
    state: DiscoveryState,
    dataset: FeatureDataset,
    config: Config,
    device: torch.device,
) -> dict[str, np.ndarray]:
    predictions = {}
    for name in tqdm(dataset.names, desc="predict test videos"):
        features = dataset.features(name)  # No test annotation is loaded here.
        logits, embeddings = model(features[None].to(device))
        reference = logits[-1, 0].argmax(0).cpu().numpy()
        predictions[name] = state.predict(
            embeddings[0].T.cpu().numpy(),
            reference,
            config.discovery,
            len(dataset.protocol.known_ids),
        )
    return predictions


def evaluate_checkpoint(
    config: Config,
    protocol: Protocol,
    training: FeatureDataset,
    testing: FeatureDataset,
    checkpoint: Path,
    device: torch.device,
    output: Path,
    discovery_path: Path | None = None,
) -> dict:
    model = load_model(checkpoint, config, protocol, device)
    fingerprint = {
        "checkpoint_sha256": checkpoint_hash(checkpoint),
        "protocol": protocol.signature(),
        "config": config.to_dict(),
        "fit_videos": training.names,
        "max_frames": training.max_frames,
    }
    if discovery_path:
        payload = torch.load(discovery_path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") != fingerprint:
            raise ValueError(
                "discovery artifact does not match this checkpoint, configuration, or training manifest"
            )
        state = DiscoveryState(payload["centroids"].numpy(), payload["bic"])
    else:
        state = fit_discovery(model, training, config, device)
        torch.save(
            {
                "format_version": 1,
                "fingerprint": fingerprint,
                "centroids": torch.from_numpy(state.centroids),
                "bic": state.bic,
            },
            output / "discovery.pt",
        )
    predictions = predict(model, state, testing, config, device)
    prediction_directory = output / "predictions"
    prediction_directory.mkdir(exist_ok=True)
    for name, prediction in predictions.items():
        np.save(prediction_directory / f"{Path(name).stem}.npy", prediction)
    targets = {
        name: testing.full_targets(name, len(prediction))
        for name, prediction in predictions.items()
    }
    result = evaluate(
        predictions,
        targets,
        protocol,
        config.ignore_labels,
        skip_empty_targets=config.protocol == "breakfast",
    )
    result.update(
        estimated_unknown_classes=len(state.centroids),
        bic=state.bic,
        checkpoint_sha256=fingerprint["checkpoint_sha256"],
    )
    save_json(output / "metrics.json", result)
    return result
