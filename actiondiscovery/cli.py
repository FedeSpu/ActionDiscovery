"""Command-line entry point; no argument parsing or logging happens on import."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from .config import load_config
from .data import FeatureDataset, Protocol
from .engine import evaluate_checkpoint, save_json, seed_everything, train


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the paper's Action Discovery pipeline"
    )
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=positive, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--discovery",
        type=Path,
        help="Reuse a compatible, training-fitted discovery artifact",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=positive)
    parser.add_argument(
        "--max-videos",
        type=positive,
        help="Smoke tests only: limit both train and test manifests",
    )
    parser.add_argument(
        "--max-frames",
        type=positive,
        help="Smoke tests only: truncate sampled sequences",
    )
    parser.add_argument(
        "--threads",
        type=positive,
        default=1,
        help="CPU PyTorch threads; increase for full runs",
    )
    parser.add_argument(
        "--wandb-project",
        help="Optional experiment tracking; otherwise logging stays local",
    )
    repository_protocols = Path(__file__).resolve().parent.parent / "protocols"
    default_protocols = (
        repository_protocols
        if repository_protocols.exists()
        else Path(sys.prefix) / "share/actiondiscovery/protocols"
    )
    parser.add_argument("--protocol-root", type=Path, default=default_protocols)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    config.validate()
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("--checkpoint is required for evaluation")
    if args.mode == "train" and (args.checkpoint or args.discovery):
        parser.error("--checkpoint and --discovery apply only to evaluation")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be fresh or empty to protect existing results")
    protocol = Protocol.load(args.protocol_root / config.protocol)
    if not 1 <= args.split <= config.splits:
        parser.error(f"--split must be between 1 and {config.splits}")
    torch.set_num_threads(args.threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; use --device cpu or repair the GPU environment"
        )
    seed_everything(config.seed)
    training = FeatureDataset(
        config,
        protocol,
        args.data_root,
        args.split,
        "train",
        args.max_videos,
        args.max_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "run.log"),
        ],
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    save_json(args.output_dir / "resolved_config.json", config.to_dict())
    save_json(
        args.output_dir / "run.json",
        {
            "mode": args.mode,
            "split": args.split,
            "device": str(device),
            "data_root": str(args.data_root.resolve()),
            "config_path": str(args.config.resolve()),
            "protocol_root": str(args.protocol_root.resolve()),
            "max_videos": args.max_videos,
            "max_frames": args.max_frames,
            "torch_version": str(torch.__version__),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        },
    )
    tracking = None
    if args.wandb_project:
        import wandb  # Optional dependency; never imported by default.

        tracking = wandb.init(
            project=args.wandb_project,
            config=config.to_dict(),
            dir=str(args.output_dir),
        )
    try:
        if args.mode == "train":
            print(train(config, training, device, args.output_dir, tracking))
        else:
            testing = FeatureDataset(
                config,
                protocol,
                args.data_root,
                args.split,
                "test",
                args.max_videos,
                args.max_frames,
            )
            result = evaluate_checkpoint(
                config,
                protocol,
                training,
                testing,
                args.checkpoint,
                device,
                args.output_dir,
                args.discovery,
            )
            print(json.dumps(result, indent=2))
            if tracking:
                tracking.log(result["metrics"])
    finally:
        if tracking:
            tracking.finish()
