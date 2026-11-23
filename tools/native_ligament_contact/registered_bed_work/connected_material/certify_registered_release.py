"""Verify native matched controls, installed material ancestry and two full-path prefix replays."""
import csv
import json
import re
from pathlib import Path
from run_bed_work import digest


def certify(root):
    def read(path):
        return json.loads((root/path).read_text())
    def trace(path):
        with (root/path).open() as f:
            return list(csv.DictReader(f))
    reference=read("installed-prior/connected-columns.json")
    matched=read("installed-matched/connected-columns.json")
    lengths=read("installed-assigned/assignment.json")
    run=read("registered-79/terminal.receipt.json")
    replay=read("launcher-79/terminal.receipt.json")
    for field in ("accepted_body_poses","accepted_step","accepted_root_fingerprint_hex","accepted_transaction_fingerprint","accepted_time_s"):
        if run[field]!=replay[field]:
            raise ValueError("Native repeats differ in "+field)
    if reference["source_identities"]["INSTALLED_TETRAHEDRA"]!=matched["source_identities"]["INSTALLED_TETRAHEDRA"]:
        raise ValueError("Different installed cells")
    if not matched["registration"]["single_rigid_transform"] or matched["float32_inversions"] or matched["loaded_state"]["float32_inversions"]:
        raise ValueError("Invalid registered tissue geometry")
    used=Path(read("registered-79/launch.json")["environment"]["NUMI_HUMAN_REGISTERED_BED_STOCK_CSV"])
    if lengths["stock"]!=digest(used) or digest(root/"installed-assigned/registered-stock.csv")!=digest(used):
        raise ValueError("The native force owner used different material sections")
    native=trace("registered-79/registered-bed-columns.csv")
    source={r["vertex"]:r for r in trace("installed-assigned/registered-stock.csv")}
    assigned={r["vertex"]:r for r in native if r["thickness_source_vertex"]=="4294967295" and r["vertex"] in source}
    if set(assigned)!=set(source) or len(assigned)!=lengths["row_count"]:
        raise ValueError("Missing native skin thickness assignment")
    import numpy as np
    for i,r in source.items():
        if not all(np.float32(r[k])==np.float32(assigned[i][k]) for k in ("skin_m","fat_m","area_m2")):
            raise ValueError("Native material output differs")
    controls=[]
    for name in ("pose-control-16","stock-control-16","recruited-16"):
        job=read(name+"/execution.json"); witness=read(name+"-source/witness.json")
        failure=read(name+"/failure.json")
        step=witness["material_step_index"]
        if job["requested_steps"]!=8000 or job["exit_code"]==0 or witness["invalid_columns_observed"]<1 or step!=failure["respiratory_accepted_status"][0]:
            raise ValueError("Unmatched rejection")
        controls.append(dict(case=name,steps_before_rejection=step,seconds_before_rejection=.002*step,trial=step+1,witness=witness))
    if controls[-1]["steps_before_rejection"]!=run["accepted_step"]:
        raise ValueError("Prefix differed from long request")
    coupled=trace("registered-79/resting-coupled.csv")
    if coupled!=trace("launcher-79/resting-coupled.csv") or any(float(r[k])!=0 for r in coupled for k in ("root_assistance_n","root_assistance_nm","body_mass_residual_kg")):
        raise ValueError("Assistance, mass, or physiology mismatch")
    pressure=[float(r["pleural_pa"]) for r in coupled]
    if max(pressure)-min(pressure)<1:raise ValueError("Variable pleural pressure inactive")
    bed=read("registered-79/terminal-bed-certificate.json")
    mass=trace("registered-79/resting-com-momentum-diagnostic.csv")
    if len({float(r["total_body_mass_kg"]) for r in mass})!=1 or bed["count_beyond_material_domain"] or bed["physical_root"]!=run["accepted_root_fingerprint_hex"]:
        raise ValueError("Wrong registered endpoint")
    env={c:read(c+"/envelope/accepted-envelope.json") for c in ("pose-one","registered-79")}
    times=[read(c+"/execution.json")["cold_wall_s"] for c in ("registered-79","launcher-79")]
    devices=re.findall(r'Apple M\d+(?: (?:Pro|Max|Ultra))?',(root/"registered-79/stdout.log").read_text())
    if not devices:raise ValueError("Unidentified native device")
    init=next(l for l in (root/"registered-79/stdout.log").read_text().splitlines() if l.startswith("resting_initialization=dynamic_release"))
    foundation=read("calibrated/foundation.json")
    return dict(parent_revision="d8c3cac28c788506545814bb0140561606b4933a",device=devices[0],
        accepted_seconds=run["accepted_time_s"],native_root=run["accepted_root_fingerprint_hex"],
        base_pack_sha256=run["base_pack_file_sha256"],binary_sha256=digest(root/"calibrated/native/numi-human-native"),
        installed_cells=matched["cells"],installed_mass=matched["mass"],
        old_registered_reference_inversions_f32=reference["float32_inversions"],
        old_registered_loaded_inversions_f32=reference["loaded_state"]["float32_inversions"],
        matched_reference_inversions_f32=matched["float32_inversions"],matched_loaded=matched["loaded_state"],
        registered_columns=len(assigned),total_positive_columns=matched["enrolled_columns"],operative_column_sha256=digest(used),
        source_area_m2=matched["projected_area_m2"],registered_area_m2=lengths["sensitivity"]["45"]["area_m2"],
        preserved_body_mass_kg=float(mass[-1]["total_body_mass_kg"]),control_and_two_updates=controls,
        cold_wall_s=times,cold_end_to_end_factors=[run["accepted_time_s"]/t for t in times],sustained_realtime_factor=None,
        native_pleural_observations_pa=pressure,maximum_observed_root_assistance_n=0,maximum_observed_root_assistance_nm=0,
        maximum_observed_mass_residual_kg=0,last_gpu_bed_support_N=float(mass[-1]["normal_force_last_physical_step_n"]),
        independent_endpoint_pressure_integral_N=bed["normal_reaction_at_endpoint_N"],
        opening_weight_N=foundation["body_weight_N"],opening_unbalanced_moment_Nm=foundation["unbalanced_xy_moment_Nm"],
        opening_recruitment=init,geometry={c:{k:r[k] for k in ("accepted_step","accepted_time_s","nonocular_pairs","skin_self_intersections")} for c,r in env.items()},
        motion=read("registered-79/shoulder-motion.json"),qualification_16s=False,
        disposition="Preserve the matched anatomical datum for future physical transfer. Reduced tissue-normal contact still fails the shoulder material domain and whole-body geometry. The dense loaded mesh was checked at initialization; its motion was not integrated in this reduced trajectory.")


if __name__=="__main__":
    p=Path(__file__).resolve().parent
    (p/"decision.json").write_text(json.dumps(certify(p),indent=2,allow_nan=False)+"\n")
