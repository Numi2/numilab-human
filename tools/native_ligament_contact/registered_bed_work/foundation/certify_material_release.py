"""Summarize the matched native controls, independent geometry and failed long release."""
import csv
import json
import sys
from pathlib import Path

from run_bed_work import digest


def certify(root):
    def read(name):
        return json.loads((root / name).read_text())

    release = read("surface-92/terminal.receipt.json")
    certificate = read("surface-92/terminal-bed-certificate.json")
    repeat = read("release-check/terminal.receipt.json")
    control = read("qualified-prefix/terminal.receipt.json")
    if any(s["accepted_body_poses"] != release["accepted_body_poses"] for s in (repeat, control)):
        raise ValueError("Presentation change altered accepted poses")
    if any(s["accepted_step"] != 92 or s["surface_audit_endpoint"] != "passed" for s in (release, repeat)):
        raise ValueError("No matched 92-step native surface")
    if (release["pack_file_sha256"] != repeat["pack_file_sha256"] or
        certificate["count_beyond_material_domain"] or certificate["checked_columns"] != 49294):
        raise ValueError("Failed endpoint reproduction or invalid material")
    if (root / "surface-16/terminal.receipt.json").exists() or read("surface-16/execution.json")["exit_code"] == 0:
        raise ValueError("Long request disagrees with retained rejection")
    fault=read("surface-16-source/witness.json")
    if fault["material_step_index"] != 92 or fault["depth_m"] <= fault["allowed_depth_m"]:
        raise ValueError("No matching material-domain rejection")
    with (root/"surface-92/resting-coupled.csv").open() as f:
        pleural = [float(r["pleural_pa"]) for r in csv.DictReader(f)]
    if max(pleural) <= min(pleural):
        raise ValueError("Pleural pressure did not vary")
    interfaces={k:read(k+"/skin-load-geometry.json")[1]["loaded_interface"] for k in ("qualified-prefix","surface-92")}
    if interfaces["surface-92"]["gaps_outside_1mm"] or not interfaces["qualified-prefix"]["gaps_outside_1mm"]:
        raise ValueError("Independent surface comparison disagrees with control")
    cases=[("opening","geometry-88/opening-envelope","opening.json"),("legacy_plane","qualified-prefix/envelope","accepted-envelope.json"),
           ("series_boundary","surface-92/envelope","accepted-envelope.json"),("dense_unloaded_opening","outer-envelope","physical-outer.json")]
    geometries={}
    for name,folder,mesh in cases:
        banks=read(folder+"/physical-banks.json")
        geometries[name]={"banks":banks["checked_banks"],"nonocular_pairs":banks["physical_nonocular_pairs"],
                          "skin_self_intersections":read(folder+"/"+mesh)["skin_self_intersections"],"bone_pairs":banks["bone_pairs"]}
    counters=read("ranked-probe.json")
    t=release["accepted_time_s"]
    walls=[read(n+"/execution.json")["cold_wall_s"] for n in ("surface-92","release-check")]
    result=dict(parent_git_revision="4fa54ac01c361acd2270993167e41b77d377be47",device="Apple M6",dt_s=.002,
            goal_16s=False,sustained_realtime_factor=None,scope="Reduced normal skin/fat/foam column contact on the existing native articulated human. Static, isotropic reference foam. Whole-body tissue equilibrium is open.",
            accepted_steps=release["accepted_step"],accepted_time_s=t,cold_wall_s=walls,cold_realtime_factors=[t/w for w in walls],
            pressure_observations_pa=pleural,force_and_material=certificate,first_invalid_trial=fault,
            overload_m=fault["depth_m"]-fault["allowed_depth_m"],matched_interface=interfaces,
            measured_source_geometry=geometries,anatomical_pack_sha256=release["pack_file_sha256"],
            binary_sha256=digest(root/"coupled-surface/numi-human-native"),
            first_step_root_translation_peak_m_s2=[counters["control_max_root_translation_acceleration_m_s2"],counters["loaded_max_root_translation_acceleration_m_s2"]],
            first_step_joint_peak_rad_s2=[counters["control_max_joint_acceleration_rad_s2"],counters["loaded_max_joint_acceleration_rad_s2"]],
            disposition="Retain the physically resolved contact branch as a reproducible experiment. Anatomical qualification fails at opening and at the accepted endpoint. Joint equilibrium and lateral tissue transfer remain open.")
    (root/"decision.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    certify(Path(sys.argv[1]).resolve())
