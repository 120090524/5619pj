from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SENTRY-Jury course project scaffold.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    runner = ExperimentRunner.from_yaml(args.config)
    summary = runner.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
