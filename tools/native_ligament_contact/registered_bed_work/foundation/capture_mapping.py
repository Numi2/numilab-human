"""Keep authenticated mappings alive for an independent comparison with the native surface."""
import argparse
import json
from pathlib import Path

from material_context import sha
from run_bed_work import run
from check_skin_load_geometry import check


def compare(case, source, opening, reader, output):
    config = json.loads((source / "launch.json").read_text())
    directory = output.with_name(output.name + "-source")
    directory.mkdir()
    root = Path(__file__).resolve().parent
    foundation = root / "foundation.json"
    config["pins"][str(foundation)] = sha(foundation)
    config["environment"]["NUMI_HUMAN_RESOLVED_BED_WITNESS"] = str(directory / "witness.json")
    (directory / "launch.json").write_text(json.dumps(config, indent=2) + "\n")

    def observe(receipt):
        refs = json.loads(receipt.read_text())["skin_source_mapping"]
        weights = Path(refs["influences"]["path"]).parent
        if Path(refs["vertex_map"]["path"]).parent != weights:
            raise ValueError("inconsistent skin export")
        check(case, opening, reader, weights)

    run(directory, root / "owner", output, 1, enabled=False, foundation=foundation, observe_endpoint=observe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "source", "opening", "reader", "output"):
        parser.add_argument(name, type=lambda x: Path(x).resolve())
    compare(**vars(parser.parse_args()))
