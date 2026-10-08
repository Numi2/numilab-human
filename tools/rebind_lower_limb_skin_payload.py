#!/usr/bin/env python3
"""Derive a bounded lower-limb NHSKIN visual candidate from exact anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from numilab_human.skin_lower_limb_anchor_rebind import rebind_registered_lower_limb_skin_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--myosim-artifact", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--input-provenance", type=Path)
    parser.add_argument("--tissue-payload", type=Path, required=True)
    parser.add_argument("--tissue-manifest", type=Path, required=True)
    parser.add_argument("--owner-comparison-report", type=Path, required=True)
    args = parser.parse_args()
    result = rebind_registered_lower_limb_skin_payload(
        source_payload=args.source_payload,
        registration_path=args.registration,
        myosim_artifact=args.myosim_artifact,
        output_directory=args.output_directory,
        input_provenance_path=args.input_provenance,
        tissue_payload_path=args.tissue_payload,
        tissue_manifest_path=args.tissue_manifest,
        owner_comparison_report_path=args.owner_comparison_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
