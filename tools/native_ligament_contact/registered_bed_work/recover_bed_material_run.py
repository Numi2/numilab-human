"""Archive and restore-check the changed M6 solver, reference columns and complete native observations."""

from pathlib import Path
from archive_cavity_trial import archive_files, sha
from fit_bed_material_to_dense import read


def retain(case):
    out = case / "finite-16"
    receipt = read(out / "terminal.receipt.json")
    cur = case / "finite-owner"
    inputs = case / "finite-source"
    if (
        not read(out / "constitutive-motion.json")["geometry"]
        == receipt["pack_file_sha256"]
    ):
        raise ValueError("Changed measured scene")
    files = {}
    for key, directory in [
        ("run", out),
        ("source", inputs),
        ("linear_material", case / "material-source"),
        ("linear_failure", case / "material-witness"),
    ]:
        for p in directory.rglob("*"):
            if (
                p.is_file()
                and "__pycache__" not in p.parts
                and "live-" not in str(p.relative_to(directory))
            ):
                files[key + "/" + str(p.relative_to(directory))] = p
    for p in cur.iterdir():
        if p.is_file() and (
            p.suffix in (".metal", ".metalinc", ".hpp", ".mm", ".json")
            or p.name == "numi-human-native"
        ):
            files["owner/" + p.name] = p
    for name in read(cur / "identities.json"):
        files["owner/" + name] = cur / name
    for name in (
        "fit_bed_material_to_dense.py",
        "fit_bed_section_curve.py",
        "verify_bed_constitutive_motion.py",
        "recover_bed_material_run.py",
        "run_bed_work.py",
    ):
        files["tools/" + name] = case / name
    archive_files(
        files,
        "finite-skin-fat-bed-" + sha(out / "terminal.receipt.json")[:16] + ".tar.zst",
        case / "finite-recovery.json",
        "Native supported-body motion, solver and M6 Metal library, generated nonlinear column responses, material compiler and masses pinned by underlying scene. Terminal and intermediate skin, tissue and body transforms and exact intersection witnesses are included.",
    )


if __name__ == "__main__":
    retain(Path.cwd())
