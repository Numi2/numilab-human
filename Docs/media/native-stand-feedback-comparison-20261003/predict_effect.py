#!/usr/bin/env python3
"""Emit the sealed effect interval for the registered Numi science study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("predict",))
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    data = args.model.read_bytes()
    model = json.loads(data)
    result = {
        "schema": "numi.science.prediction.v1",
        "model_sha256": hashlib.sha256(data).hexdigest(),
        "prediction": model["prediction"],
        "unit": "m",
        "observable": "root_horizontal_drift_m",
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
