"""Exercise affine-motion preservation and rejection of a manufactured floor crossing."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from prolong_skin_trial import CONTROL, digest, run

FLOOR_KEY = "NUMI_H706_FRACTIONAL_SOURCE330_NODES"


def shifted(source, destination, dz):
    with source.open(newline="") as inp, destination.open("x", newline="") as out:
        reader = csv.DictReader(inp)
        writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["current_z_m"] = format(float(row["current_z_m"]) + dz, ".17g")
            writer.writerow(row)


def main():
    configuration = json.loads((CONTROL / "launch.json").read_text())["environment"]
    source = Path(configuration[FLOOR_KEY])
    assert digest(source) == configuration[FLOOR_KEY + "_SHA256"]
    with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        directory = Path(tmp)
        evidence = {}
        for label, displacement in (("uniform_up", 0.001), ("uniform_down", -0.001)):
            table = directory / f"{label}.csv"
            report = directory / f"{label}.json"
            shifted(source, table, displacement)
            run(table, digest(table), report, publish_nodes=(displacement < 0.0))
            metrics = json.loads(report.read_text())
            assert np.isclose(metrics["initial_max_displacement_m"], abs(displacement),
                              rtol=0.0, atol=5.0e-8)
            assert metrics["inverted_cells"] == 0
            assert abs(metrics["minimum_reference_relative_det_ratio"] - 1.0) < 1.0e-7
            assert metrics["maximum_shared_interface_disagreement_m"] < 1.0e-11
            assert (metrics["bed_penetrating_outer_nodes"] > 0) == (displacement < 0.0)
            assert not report.with_suffix(".node").exists()
            evidence[label] = metrics
        output = Path(__file__).with_name("prolongation-tests.json")
        with output.open("x") as out:
            json.dump(evidence, out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    main()
