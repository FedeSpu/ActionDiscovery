"""Validate published manifests against local features and annotations without writing data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actiondiscovery.config import load_config
from actiondiscovery.data import FeatureDataset, Protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--all-features", action="store_true", help="Also load all features and annotation contents")
    args = parser.parse_args()
    config = load_config(args.config)
    protocol = Protocol.load(Path(__file__).resolve().parents[1] / "protocols" / config.protocol)
    for split in range(1, config.splits + 1):
        datasets = [FeatureDataset(config, protocol, args.data_root, split, mode) for mode in ["train", "test"]]
        if set(datasets[0].names) & set(datasets[1].names):
            raise ValueError(f"split {split}: train/test overlap")
        for dataset in datasets:
            for name in dataset.names:
                for path in [dataset.directory / "features" / f"{Path(name).stem}.npy", dataset.directory / "groundTruth" / name]:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                if args.all_features:
                    dataset.training_sample(name)
        print(f"split {split}: {len(datasets[0])} train, {len(datasets[1])} test; paths valid")


if __name__ == "__main__":
    main()
