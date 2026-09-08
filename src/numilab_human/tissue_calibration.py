"""Pinned experimental cartilage import and conditional finite-hold fitting.

The only supported cohort is three unconfined-compression repeats of the SAME
oks003 patellar plug. This is neither a population calibration nor a native
material qualification. No literature-average or synthetic curves enter this
command. The raw archive is CC BY 4.0, Chokhandre and Erdemir (2020), dataset
10.18735/WTJZ-N328, article 10.1016/j.jmbbm.2020.104025.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .model import ImportError as HumanImportError

SCHEMA = "HumanPack.tissue-calibration-candidate.v1"
BASE = "http://archive.simtk.org/oks/tissue/oks003/cartilage/"
GF_TO_N = 0.00980665
POISSON_ASSUMPTION = 0.45
DIAMETER_M_ASSUMPTION = 0.005
FINAL_WINDOW_S = 300.0
TRAIN = ("oks003-PTC-MCXX-01-01", "oks003-PTC-MCXX-01-03")
HELD_OUT = ("oks003-PTC-MCXX-01-05",)

# Acquisition manifest frozen before fitting or looking at held-out responses.
# OTMS retains its unrounded mean; the README rounds repeat 05 to 2.694 mm.
SAMPLES = {
    TRAIN[0]: (141475308, "3f869e726f8c24daa3bb01f7a0c90dd3e7e38cc4c02362b6ebc2978c53c8cc1d",
               "2017-11-13-111131-0000", 450, "fad28920477c25958fa59cab8f5c0ce2ac03af168a8cd23c262b0986efe02625"),
    TRAIN[1]: (148870403, "26cbb0dd2c5674ca43fe658c8de0324f018a6de609c9426f84918fc134435a12",
               "2017-11-14-093725-0000", 447, "63122801f1fd2e19a47d68f186b08fa0f1f7d1d51dc14ede7e5b4569120c4151"),
    HELD_OUT[0]: (147256514, "d29e1be495b5ba418460256cc1ab2af64b72db56822a547ee8c7dd51cc77ec45",
                  "2017-11-15-105543-0000", 473, "0c3d51a1e753af3bf13f8d5ce4a23c22f98a6ebebc3cabef94297ddbd3f13624"),
}
SUPPORT = (
    ("cartilage-readme.txt", "readme.txt", 9175, "fdfcd3875e83cc16b102998c500c2626e5c033c688bb3c2c5fa6e70b02f95fe7"),
    ("cartilage-license.txt", "license.txt", 236, "1abf0c1bbc0a81de9322026d27551e431aed6d9bed2d25b6ad70edc3eb377006"),
)
COMMANDS = (
    "Zero Load", "Find Contact", "Wait", "Move Relative", "Move Absolute",
    "Move Relative", "Move Relative", "Move Relative", "Sinusoid",
    "Move Relative", "Move Relative", "Move Relative", "Move Relative",
    "Zero Load", "Find Contact", "Wait", "Move Relative", "Move Absolute",
    "Move Relative", "Wait", "Wait", "Wait", "Wait",
    "Move Relative", "Wait", "Wait", "Wait", "Wait",
    "Move Relative", "Wait", "Wait", "Wait", "Wait", "Move Relative",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sources() -> list[dict]:
    entries = []
    for sample, (size, sha, suffix, xml_size, xml_sha) in SAMPLES.items():
        xml = f"{sample}_{suffix}.xml"
        entries.extend([
            dict(file=f"{sample}.txt", url=f"{BASE}Mach1/{sample}/data.txt", bytes=size, sha256=sha),
            dict(file=xml, url=f"{BASE}OTMS/{sample}/{xml}", bytes=xml_size, sha256=xml_sha),
        ])
    entries.extend(dict(file=name, url=BASE + remote, bytes=size, sha256=sha)
                   for name, remote, size, sha in SUPPORT)
    return entries


def verify_file(path: Path, expected: dict) -> None:
    if path.stat().st_size != expected["bytes"]:
        raise HumanImportError(f"source size mismatch: {path.name}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    if result.hexdigest() != expected["sha256"]:
        raise HumanImportError(f"source SHA256 mismatch: {path.name}")


def fetch_sources(directory: Path) -> None:
    """Fetch only the eight pinned objects; reject partial or changed content."""
    directory.mkdir(parents=True, exist_ok=True)
    for entry in sources():
        target = directory / entry["file"]
        if target.exists():
            verify_file(target, entry)
            continue
        with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".partial",
                                         dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
        start, count = time.monotonic(), 0
        try:
            with urllib.request.urlopen(entry["url"], timeout=20) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    count += len(chunk)
                    if count > entry["bytes"] or time.monotonic() - start > 300:
                        raise HumanImportError(f"bounded source download exceeded: {target.name}")
                    stream.write(chunk)
            verify_file(temporary, entry)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)



@dataclass
class Block:
    command: str
    header: dict[str, str]
    count: int
    first: tuple[float, float, float]
    last: tuple[float, float, float]
    rows: list[tuple[float, float, float]]


def parse_mach1(path: Path) -> list[Block]:
    """Stream all rows; retain full SR ramps/holds, not the 1,000-cycle signal.

    No resampling or smoothing. Every raw row must be finite, complete and in
    increasing segment-local time. SHA validation belongs to the caller so the
    parser can also be exercised with explicitly synthetic format fixtures.
    """
    blocks: list[Block] = []
    state, command, header = "start", "", {}
    first = last = None
    rows: list[tuple[float, float, float]] = []
    count = 0
    with path.open("r", encoding="ascii", newline=None) as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if len(raw) > 4096:
                raise HumanImportError(f"oversized Mach1 line {number}")
            if state == "start":
                if line != "<Mach-1 File>":
                    raise HumanImportError("expected Mach1 file header")
                state = "metadata"
            elif state == "units":
                if line != "Time, s\tPosition (z), mm\tFz, gf":
                    raise HumanImportError("Mach1 units must be seconds, mm and gf")
                state = "data"
            elif state == "data":
                if line == "<END DATA>":
                    if not count or first is None or last is None:
                        raise HumanImportError("empty Mach1 data block")
                    blocks.append(Block(command, header, count, first, last, rows))
                    state, command, header = "metadata", "", {}
                    first = last = None
                    rows, count = [], 0
                    continue
                try:
                    row = tuple(float(field) for field in line.split("\t"))
                except ValueError as error:
                    raise HumanImportError(f"invalid Mach1 row {number}") from error
                if len(row) != 3 or not all(math.isfinite(value) for value in row):
                    raise HumanImportError(f"nonfinite/incomplete Mach1 row {number}")
                if row[0] < 0 or (last is not None and row[0] <= last[0]):
                    raise HumanImportError("Mach1 time must strictly increase within each block")
                if first is None:
                    if row[0] != 0:
                        raise HumanImportError("Mach1 block must start at local time zero")
                    first = row
                last = row
                count += 1
                if count > 2_000_000:
                    raise HumanImportError("Mach1 block exceeds pinned protocol bounds")
                if 18 <= len(blocks) <= 32:
                    rows.append(row)
            elif line == "<DATA>":
                if len(blocks) >= len(COMMANDS) or command != COMMANDS[len(blocks)]:
                    raise HumanImportError(f"unexpected Mach1 command {len(blocks)}: {command}")
                state = "units"
            elif line == "<INFO>":
                state = "info"
            elif state == "info":
                if line == "<END INFO>":
                    state = "metadata"
            elif line.startswith("<"):
                if command:
                    raise HumanImportError("Mach1 command has no data")
                command = line.removeprefix("<").removesuffix(">")
            elif line and command and "\t" in line:
                key, value = line.split("\t", 1)
                if key in header:
                    raise HumanImportError("duplicate Mach1 command field")
                header[key] = value
            elif line and not command:
                raise HumanImportError("unexpected trailing Mach1 content")
    if state != "metadata" or command or len(blocks) != len(COMMANDS):
        raise HumanImportError("truncated/incomplete Mach1 protocol")
    return blocks


def thickness(path: Path, sample: str) -> float:
    try:
        root = ET.parse(path).getroot()
        if (root.tag != "OTMS" or
                not root.findtext("Sample", "").startswith(sample + "_") or
                root.find("Statistics").get("units") != "mm" or
                root.find("Thickness").get("units") != "mm"):
            raise ValueError("OTMS identity or units mismatch")
        values = json.loads(root.findtext("Thickness"))
        if not (isinstance(values, list) and len(values) >= 5 and
                all(type(value) in (float, int) and math.isfinite(value) and 0 < value < 10 for value in values)):
            raise ValueError("OTMS thickness values are invalid")
        average = float(root.findtext("Statistics/average"))
        if not (math.isfinite(average) and math.isclose(average, sum(values) / len(values), abs_tol=1e-12)):
            raise ValueError("OTMS thickness average disagrees with measurements")
        return average * 0.001
    except (AttributeError, ValueError, TypeError, ET.ParseError) as error:
        raise HumanImportError(f"invalid or mismatched OTMS thickness: {path.name}") from error


def time_mean(rows: list[tuple[float, float, float]], window: float, column: int) -> float:
    """Trapezoidal mean over an exact trailing physical-time window."""
    if len(rows) < 2 or window <= 0 or not math.isfinite(window):
        raise HumanImportError("invalid time averaging window")
    end, start = rows[-1][0], rows[-1][0] - window
    if rows[0][0] > start:
        raise HumanImportError("hold does not cover averaging window")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        if right[0] <= left[0]:
            raise HumanImportError("hold times must increase")
        if right[0] <= start:
            continue
        t = max(start, left[0])
        y = left[column] + (right[column] - left[column]) * (t - left[0]) / (right[0] - left[0])
        area += 0.5 * (y + right[column]) * (right[0] - t)
    return area / (end - start)


def observations(blocks: list[Block], height_m: float) -> dict:
    if len(blocks) != 34 or tuple(b.command for b in blocks) != COMMANDS:
        raise HumanImportError("incomplete relaxation protocol")
    if not math.isfinite(height_m) or not 0 < height_m < 0.01:
        raise HumanImportError("invalid specimen height")
    if blocks[8].header.get("Number of Cycles:") != "1000" or blocks[8].header.get("Freqency, Hz:") != "2.00000":
        raise HumanImportError("unexpected preconditioning protocol")
    for index in (1, 14):
        if blocks[index].header.get("Stop criteria, gf:") != "10.000000":
            raise HumanImportError("unexpected contact threshold")
    # The archive protocol owns the strain reference. Its Move Absolute
    # command positions the platen exactly 0.300 mm before the contact
    # reference. A new force crossing during the FAST relaxation ramp would
    # silently change that reference (about 3% strain in repeat01).
    try:
        reference = float(blocks[17].header["Position, mm:"]) + 0.300
    except (KeyError, ValueError) as error:
        raise HumanImportError("missing protocol contact reference") from error
    if not math.isfinite(reference) or any(
            abs(blocks[i].last[1] - reference) > 0.002 for i in (14, 15)):
        raise HumanImportError("protocol reference disagrees with measured contact position")
    points = []
    for ramp_index in (18, 23, 28):
        group = blocks[ramp_index + 1:ramp_index + 5]
        for block, expected in zip(group, (10, 100, 1000, 690)):
            stamp = block.header.get("Wait:", "")
            fields = stamp.split(":")
            if len(fields) != 3 or sum(float(v) * f for v, f in zip(fields, (3600, 60, 1))) != expected:
                raise HumanImportError("unexpected stress-relaxation hold duration")
            if not expected <= block.last[0] <= expected + 1:
                raise HumanImportError("incomplete stress-relaxation hold")
        final = group[-1].rows
        position = time_mean(final, FINAL_WINDOW_S, 1)
        force = time_mean(final, FINAL_WINDOW_S, 2) * GF_TO_N
        strain = (position - reference) * 0.001 / height_m
        expected_strain = 0.05 * (1 + (ramp_index - 18) // 5)
        if abs(strain - expected_strain) > 0.003 or force <= 0:
            raise HumanImportError("invalid compressive finite-hold observation")
        if max(row[1] for row in final) - min(row[1] for row in final) > 0.002:
            raise HumanImportError("final hold did not maintain displacement")
        points.append(dict(
            ramp_block=ramp_index, final_hold_block=ramp_index + 4,
            compression_strain=strain, force_N=force,
            nominal_stress_Pa=force / (math.pi * DIAMETER_M_ASSUMPTION**2 / 4),
            hold_elapsed_s=sum(block.last[0] for block in group),
            final_window_s=FINAL_WINDOW_S, final_hold_rows=len(final),
            final_hold_fractional_force_drop=(final[0][2] - final[-1][2]) * GF_TO_N / force,
            final_hold_force_N=[[row[0], row[2] * GF_TO_N] for row in final],
        ))
    return dict(height_m=height_m, reference_position_mm=reference,
                reference_force_N=10 * GF_TO_N, points=points)


def nh_force_basis(strain: float, poisson: float = POISSON_ASSUMPTION) -> float:
    """Nominal force per Pa of mu, zero lateral traction, compressible NH.

    For axial stretch a, lateral stretch squared q solves
    q - 1 + (lambda/mu) log(a*q) = 0. Thus -Pzz/mu = q/a-a.
    This is the same log-J NH energy as the emitted native candidate.
    """
    if not math.isfinite(strain) or not 0 < strain < 1 or not 0 <= poisson < 0.5:
        raise HumanImportError("invalid uniaxial NH state")
    a, r = 1 - strain, 2 * poisson / (1 - 2 * poisson)
    low, high = 1.0, 1.0 / a
    for _ in range(80):
        q = (low + high) / 2
        if q - 1 + r * math.log(a * q) > 0:
            high = q
        else:
            low = q
    q = (low + high) / 2
    return math.pi * DIAMETER_M_ASSUMPTION**2 / 4 * (q / a - a)


def fit_mu(training: Iterable[dict]) -> float:
    points = list(training)
    if not points:
        raise HumanImportError("no training observations")
    basis = [nh_force_basis(point["compression_strain"]) for point in points]
    force = [point["force_N"] for point in points]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in force):
        raise HumanImportError("invalid training force")
    return sum(x * y for x, y in zip(basis, force)) / sum(x * x for x in basis)


def metrics(points: list[dict], mu: float) -> dict:
    residuals = [mu * nh_force_basis(p["compression_strain"]) - p["force_N"] for p in points]
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    measured_rms = math.sqrt(sum(p["force_N"]**2 for p in points) / len(points))
    return dict(observations=len(points), force_rmse_N=rms,
                force_nrmse_relative_to_measured_rms=rms / measured_rms,
                maximum_absolute_force_error_N=max(abs(r) for r in residuals),
                signed_force_errors_N=residuals)


def material_text(mu: float, source_fingerprint: str) -> str:
    # No identifiable keywords: the fitted coefficient is conditional on
    # assumed Poisson ratio and a preloaded proxy reference configuration.
    lame = mu * 2 * POISSON_ASSUMPTION / (1 - 2 * POISSON_ASSUMPTION)
    return f"""// UNQUALIFIED finite-hold same-plug candidate; not a native calibration certificate.
// Source bundle SHA256: {source_fingerprint}
// CC BY 4.0; Chokhandre and Erdemir; dataset 10.18735/WTJZ-N328.
// Fit: {', '.join(TRAIN)}. Held out: {', '.join(HELD_OUT)}.
// Assumed nu=0.45 and density=1000; loaded 10gf proxy reference, not stress free.
// No relaxation/permeability identification; see calibration-candidate.json.
material oks003_patella_finite_hold_unqualified {{
    parameter density : kg/m^3 = 1000 in [900, 1300];
    parameter mu : Pa = {mu:.17g} in [1, 1e8];
    parameter lambda : Pa = {lame:.17g} in [1, 1e9];
    model neo_hookean;
    energy = neo_hookean(mu, lambda);
    valid = J() - 0.5;
    supports fem;
    interface {{
        static_friction = 0.0;
        dynamic_friction = 0.0;
        restitution = 0.0;
        adhesion = 0.0;
    }}
    limits {{
        minimum_J = 0.5;
        maximum_J = 1.75;
        maximum_stress = 1e8;
        maximum_energy_density = 5e7;
    }}
}}
"""


def calibrate(directory: Path, output: Path) -> dict:
    entries = sources()
    for entry in entries:
        verify_file(directory / entry["file"], entry)
    records = {}
    for sample, (_, _, suffix, _, _) in SAMPLES.items():
        blocks = parse_mach1(directory / f"{sample}.txt")
        height = thickness(directory / f"{sample}_{suffix}.xml", sample)
        records[sample] = dict(role="training" if sample in TRAIN else "held_out",
                               **observations(blocks, height),
                               block_row_counts=[block.count for block in blocks])
    training = [p for sample in TRAIN for p in records[sample]["points"]]
    held_out = [p for sample in HELD_OUT for p in records[sample]["points"]]
    mu = fit_mu(training)
    fingerprint = digest(entries)
    material = material_text(mu, fingerprint).encode()
    report = dict(
        schema=SCHEMA, status="unqualified_finite_hold_candidate", qualified=False,
        sources=entries, source_bundle_sha256=fingerprint,
        importer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_doi="10.18735/WTJZ-N328", article_doi="10.1016/j.jmbbm.2020.104025",
        license="CC-BY-4.0", donor="oks003", physical_plug="oks003-PTC-MCXX-01",
        scope="three unconfined compression repeats of one patellar cartilage plug",
        split=dict(training=list(TRAIN), held_out=list(HELD_OUT),
                   unit="test day within the same physical plug", frozen_before_fit=True,
                   population_holdout=False),
        preprocessing=dict(raw_units=dict(time="s", position="mm", force="gf"),
                           gf_to_N=GF_TO_N, mm_to_m=0.001,
                           reference="post-preconditioning protocol Move Absolute position plus its declared 0.300 mm offset; checked against Find Contact and settled Wait positions",
                           nominal_strain_gate="5,10,15 percent within 0.003 absolute strain",
                           force_baseline="instrument zero; no subtraction of 10 gf preload",
                           averaging="trapezoidal physical-time average of last 300 s of final 690 s hold block",
                           filtering="none", equilibrium_assumed=False),
        assumptions=dict(poisson_ratio=POISSON_ASSUMPTION, density_kg_m3=1000,
                         diameter_m=DIAMETER_M_ASSUMPTION, homogeneous_isotropic=True,
                         lateral_boundary="zero traction", platen_friction=0,
                         zero_stress_reference="unknown; protocol contact configuration is a preloaded proxy"),
        fit=dict(objective="unweighted squared force error over six training finite-hold points",
                 mu_Pa=mu, lame_lambda_Pa=9 * mu, bulk_modulus_Pa=(9 + 2/3) * mu,
                 coefficient_status="conditional apparent coefficient; not independently identified intrinsic material",
                 training=metrics(training, mu), held_out=metrics(held_out, mu)),
        unidentifiable=["stress-free reference geometry/prestress", "Poisson ratio/bulk modulus",
                        "density", "anisotropy/fibres", "permeability/poroelasticity",
                        "physical viscosity/relaxation spectrum", "platen friction"],
        not_qualified=["equilibrium", "native specimen boundary-value response", "other loading modes",
                       "whole patellar cartilage", "preload-consistent stress-free parameter identification", "other donors or locations", "costal cartilage",
                       "ligaments/tendons", "right-knee mirror", "whole-human mechanics"],
        samples=records, anatomical_target=dict(source_region="PTC", specimen_side="left", donor="oks003",
                               binding="same donor and named patellar cartilage; plug location is not a whole-region material assignment",
                               production_force_owner_fraction=0),
        native_material=dict(file="oks003-patella-finite-hold-unqualified.nmatter",
                                            sha256=hashlib.sha256(material).hexdigest(),
                                            compiled=False, solver_validated=False),
    )
    report["report_sha256"] = digest(report)
    output.mkdir(parents=True, exist_ok=True)
    (output / report["native_material"]["file"]).write_bytes(material)
    (output / "calibration-candidate.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def command(arguments: argparse.Namespace) -> int:
    if arguments.fetch:
        fetch_sources(arguments.sources)
    report = calibrate(arguments.sources, arguments.output)
    print(json.dumps(dict(status=report["status"], qualified=False, fit=report["fit"],
                          report=str(arguments.output / "calibration-candidate.json")), indent=2))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sources", type=Path, required=True, help="directory of the eight pinned raw/OTMS/license objects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="download only the pinned CC-BY-4.0 source objects")
    parser.set_defaults(handler=command)
