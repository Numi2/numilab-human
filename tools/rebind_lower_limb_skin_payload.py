#!/usr/bin/env python3
"""Deprecated CLI: per-owner NHSKIN rebinding violates the shared atlas."""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.error(
        "individual lower-limb binding replacement is not production-qualified: "
        "NHSKIN uses one shared atlas frame. Use a source-registered geometry "
        "candidate that preserves all 86 canonical binding records."
    )


if __name__ == "__main__":
    raise SystemExit(main())
