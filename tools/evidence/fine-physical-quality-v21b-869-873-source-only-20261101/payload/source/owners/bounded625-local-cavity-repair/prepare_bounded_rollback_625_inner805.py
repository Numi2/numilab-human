from __future__ import annotations
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, sys
import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
TASK = D / "bounded625-local-cavity-repair"
HELPER = TASK / "bounded_point_component_rollback_625.py"
BASE_DRIVER = D / "fixed-boundary-quality-topology-196/run_fixed_boundary_dp_batch_197.py"
QUALITY_OWNER = D / "fixed_boundary_volume_flip_198_root.py"
QUALITY_OWNER_SHA = "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
ROLE_ORDER = ("D", "O", "I")

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_json(path: Path):
    return json.loads(path.read_text())

def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}

def save_npz(path: Path, payload):
    np.savez_compressed(path, **payload)

def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def entry(path: Path):
    return {"path": str(path.resolve()), "sha256": sha(path)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    state_path=a.state.resolve(); config_path=a.config.resolve(); audit_path=a.audit.resolve()
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    state = load_json(state_path)
    cfg = load_json(config_path)
    audit = load_json(audit_path)
    if audit.get("config",{}).get("path")!=str(config_path) or audit.get("config",{}).get("sha256")!=sha(config_path):
        raise ValueError("exact 805 report does not bind supplied candidate config")
    source_ref=state.get("source_state",{})
    if not isinstance(source_ref,dict) or not {"path","sha256"}.issubset(source_ref):
        raise ValueError("candidate state lacks a pinned parent state")
    source_state_path=Path(source_ref["path"]).resolve()
    if sha(source_state_path)!=source_ref["sha256"]:
        raise ValueError("candidate source-state parent hash mismatch")
    source_state=load_json(source_state_path)
    helper = import_module("bounded_point_rollback_625_run", HELPER)
    baseline, trial = {}, {}
    for role in ROLE_ORDER:
        be = cfg["baseline_candidates"][role]
        ce = cfg["candidate_surfaces"][role]
        if sha(Path(be["path"])) != be["sha256"] or sha(Path(ce["path"])) != ce["sha256"]:
            raise ValueError(f"{role}: surface input pin mismatch")
        if state["source_surfaces"][role] != ce:
            raise ValueError(f"{role}: candidate state/config surface pin mismatch")
        if source_state.get("source_surfaces",{}).get(role)!=be:
            raise ValueError(f"{role}: candidate baseline surface differs from parent state")
        baseline[role] = load_npz(Path(be["path"]))
        trial[role] = load_npz(Path(ce["path"]))

    base197=import_module("rollback_face_id_owner_197",BASE_DRIVER)
    baseline_face_max=max(int(base197.FACE_BASE[r])+len(np.asarray(baseline[r]["faces"],np.int64))-1 for r in ROLE_ORDER)
    wall_face_max=max(900000+90-1,901000+90-1)
    face_id_base=max(baseline_face_max,wall_face_max)+1_000_000
    if audit.get("all185_receipt",{}).get("negative_face_index")!="global index is -1 minus face_id; query faces must have nonnegative IDs":
        raise ValueError("unexpected all185 face-ID convention")
    surface_plan = helper.build_rollback_plan(audit, baseline, trial, changed_face_id_base=face_id_base)
    baseline_paths = {"Skin":Path(source_state["Skin_current"]["path"]), "Fat":Path(source_state["Fat_current"]["path"])}
    trial_paths = {"Skin":Path(state["Skin_current"]["path"]), "Fat":Path(state["Fat_current"]["path"])}
    for vrole, path in baseline_paths.items():
        if sha(path) != source_state[vrole + "_current"]["sha256"]:
            raise ValueError(vrole + " baseline volume pin mismatch")
    for vrole, path in trial_paths.items():
        if sha(path) != state[vrole + "_current"]["sha256"]:
            raise ValueError(vrole + " trial volume pin mismatch")
    volume_baseline = {r:load_npz(p) for r,p in baseline_paths.items()}
    volume_trial = {r:load_npz(p) for r,p in trial_paths.items()}

    plan = helper.expand_rollback_over_free_node_components(
        surface_plan, baseline, trial, volume_baseline, volume_trial
    )
    final_surfaces = helper.restore_surface_coordinates(baseline, trial, plan)
    for role in ROLE_ORDER:
        bp = helper.point_array(baseline[role], role + " baseline")
        tp = helper.point_array(trial[role], role + " trial")
        fp = helper.point_array(final_surfaces[role], role + " final")
        rolled = np.asarray(plan["rollback_point_rows_by_role"][role], dtype=np.int64)
        safe = np.asarray(plan["retained_safe_point_rows_by_role"][role], dtype=np.int64)
        if len(rolled) and not np.array_equal(fp[rolled].view(np.uint32), bp[rolled].view(np.uint32)):
            raise ValueError(f"{role}: rollback surface coordinates are not baseline-identical")
        if len(safe) and not np.array_equal(fp[safe].view(np.uint32), tp[safe].view(np.uint32)):
            raise ValueError(f"{role}: safe moved coordinates were not preserved")
        for key in trial[role]:
            if key not in ("points_world_m_f32", "points_world_m") and not np.array_equal(final_surfaces[role][key], trial[role][key]):
                raise ValueError(f"{role}: non-coordinate surface payload changed: {key}")

    volume_final = {}
    volume_coordinate_rows = {}
    for vrole in ("Skin", "Fat"):
        volume_final[vrole], volume_coordinate_rows[vrole] = helper.restore_volume_coordinate_rows(
            volume_baseline[vrole], volume_trial[vrole], baseline, trial, final_surfaces, plan, volume_role=vrole
        )

    pred = import_module("point_rollback_quality_owner_198", QUALITY_OWNER)
    if sha(QUALITY_OWNER) != QUALITY_OWNER_SHA:
        raise ValueError("native quality owner implementation SHA mismatch")
    quality = {}
    quality_owner_sha = QUALITY_OWNER_SHA
    for vrole in ("Skin", "Fat"):
        check = helper.verify_native_nonregression(pred, volume_baseline[vrole], volume_final[vrole])
        bp = helper.point_array(volume_baseline[vrole], vrole + " baseline")
        tp = helper.point_array(volume_trial[vrole], vrole + " trial")
        fp = helper.point_array(volume_final[vrole], vrole + " final")
        tet = np.asarray(volume_final[vrole]["tetrahedra"], dtype=np.int64)
        if not np.array_equal(tet, np.asarray(volume_baseline[vrole]["tetrahedra"], dtype=np.int64)):
            raise ValueError(vrole + ": volume topology differs from baseline")
        bm = np.asarray(volume_baseline[vrole]["material_indices"])
        tm = np.asarray(volume_trial[vrole]["material_indices"])
        fm = np.asarray(volume_final[vrole]["material_indices"])
        if not np.array_equal(bm,tm) or not np.array_equal(bm,fm):
            raise ValueError(vrole + ": material rows differ from baseline")
        bd,bq = pred.measurements(bp,tet)
        td,tq = pred.measurements(tp,tet)
        fd,fq = pred.measurements(fp,tet)
        bg = np.asarray(pred.passes(bd,bq),bool)
        tg = np.asarray(pred.passes(td,tq),bool)
        fg = np.asarray(pred.passes(fd,fq),bool)
        per_material={}
        for mat in np.unique(bm):
            rows=(bm==mat)
            base_bad=int(np.count_nonzero(~bg[rows]))
            trial_bad=int(np.count_nonzero(~tg[rows]))
            final_bad=int(np.count_nonzero(~fg[rows]))
            regressions=int(np.count_nonzero(bg[rows]&~fg[rows]))
            if regressions or final_bad>base_bad:
                raise ValueError(f"{vrole} material {int(mat)} exceeds baseline quality or regresses: {base_bad}->{final_bad}, regressions={regressions}")
            per_material[str(int(mat))]={
                "cells":int(np.count_nonzero(rows)),
                "baseline_failures":base_bad,
                "trial_648_failures":trial_bad,
                "final_failures":final_bad,
                "baseline_good_regressions":regressions,
            }
        if np.any(fd<=0): raise ValueError(vrole+": nonpositive Float32 determinant after rollback")
        quality[vrole]={
            **check,
            "baseline_failures":int(np.count_nonzero(~bg)),
            "trial_648_failures":int(np.count_nonzero(~tg)),
            "rollback_final_failures":int(np.count_nonzero(~fg)),
            "rollback_quality_failures_added_vs_trial":int(np.count_nonzero(~fg)-np.count_nonzero(~tg)),
            "per_material":per_material,
        }

    out.mkdir(parents=True)
    surface_entries={}
    for role in ROLE_ORDER:
        path=out/f"candidate-{role}-typed-current.npz"
        save_npz(path,final_surfaces[role]); surface_entries[role]=entry(path)
    volume_entries={}
    for vrole in ("Skin","Fat"):
        path=out/f"{vrole.lower()}-coupled-quality-point-rollback-proposal.npz"
        save_npz(path,volume_final[vrole]);volume_entries[vrole]=entry(path)
    plan_path=out/"rollback-plan.json"
    plan_path.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
    plan_entry=entry(plan_path)

    changed_roles=[r for r in ROLE_ORDER if len(plan["retained_safe_point_rows_by_role"][r])>0]
    if not changed_roles: raise ValueError("rollback removed all bounded point changes; no candidate audit is needed")
    next_cfg=copy.deepcopy(cfg)
    next_cfg["candidate_surfaces"]=surface_entries
    next_cfg["changed_roles"]=changed_roles
    next_cfg["bounded_point_repair"]={r:cfg["bounded_point_repair"][r] for r in changed_roles}
    config_out=out/"context-config.json"
    config_out.write_text(json.dumps(next_cfg,indent=2,sort_keys=True)+"\n")

    next_state=copy.deepcopy(state)
    next_state["source_surfaces"]=surface_entries
    for role in ROLE_ORDER: next_state["source_surfaces"][role]=surface_entries[role]
    for vrole in ("Skin","Fat"):
        next_state[vrole+"_current"]["path"]=volume_entries[vrole]["path"]
        next_state[vrole+"_current"]["sha256"]=volume_entries[vrole]["sha256"]
        v=volume_final[vrole];points=helper.point_array(v,vrole);t=np.asarray(v["tetrahedra"],np.int64)
        det,q=pred.measurements(points,t)
        next_state[vrole+"_current"].update({
            "nodes":int(len(points)),"cells":int(len(t)),
            "native_quality_failures":int(np.count_nonzero(~np.asarray(pred.passes(det,q),bool))),
            "nonpositive":int(np.count_nonzero(det<=0)),
            "missing_target_faces":"PENDING_805","extra_actual_faces":"PENDING_805",
        })
    next_state["surface_crossings"]="PENDING_SERIAL_805_REAUDIT"
    next_state["physical_admission"]=False
    next_state["native_steps"]=0
    next_state["witness_component_rollback"]={
        "status":"PREPARED_FOR_SERIAL_805_REAUDIT",
        "parent_state":entry(state_path),"parent_config":entry(config_path),"failed_805_audit":entry(audit_path),
        "rollback_plan":plan_entry,"candidate_surfaces":surface_entries,"candidate_volumes":volume_entries,
        "face_id_base":face_id_base,"quality_owner_sha256":quality_owner_sha,
        "native_quality":quality,
        "volume_coordinate_rows":{
            r:{"affected_volume_point_rows":x["affected_volume_point_rows"]} for r,x in volume_coordinate_rows.items()
        },
        "expanded_component_closure":plan["component_closure"],
    }
    state_out=out/"state.json"
    state_out.write_text(json.dumps(next_state,indent=2,sort_keys=True)+"\n")
    manifest={
        "status":"PASS_WITNESS_COMPONENT_ROLLBACK_PREPARED_FOR_EXACT_AUDIT",
        "inputs":{
            "state":entry(state_path),"config":entry(config_path),"failed_805_audit":entry(audit_path),
            "source_baseline_state":entry(source_state_path),"helper":entry(HELPER),
            "quality_owner_sha256":quality_owner_sha,
        },
        "output_state":entry(state_out),"output_config":entry(config_out),
        "rollback_plan":plan_entry,"candidate_surfaces":surface_entries,"candidate_volumes":volume_entries,
        "witness_count":len(plan["witnesses"]),
        "surface_rollback_point_counts":{r:len(plan["rollback_point_rows_by_role"][r]) for r in ROLE_ORDER},
        "surface_safe_point_counts":{r:len(plan["retained_safe_point_rows_by_role"][r]) for r in ROLE_ORDER},
        "volume_rollback_point_counts":{r:len(plan["rollback_volume_point_rows_by_role"][r]) for r in ("Skin","Fat")},
        "quality":quality,"physical_admission":False,"native_steps":0,
    }
    manifest_path=out/"rollback-preparation-receipt.json"
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()

