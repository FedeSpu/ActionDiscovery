"""Average metrics.json files for all folds of a single protocol."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from actiondiscovery.metrics import summarize_folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics",
        type=Path,
        nargs="+",
        help="one metrics.json per fold, from the same protocol",
    )
    args = parser.parse_args()
    if len({path.resolve() for path in args.metrics}) != len(args.metrics):
        parser.error("duplicate metrics files")
    results = [json.loads(path.read_text()) for path in args.metrics]
    print(json.dumps(summarize_folds(results), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
