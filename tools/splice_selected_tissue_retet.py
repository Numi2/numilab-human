#!/usr/bin/env python3
"""Build a source-preserving full-mesh candidate from a selected-cell retet packet.

This is an authoring/assembly step only. It preserves all outside-used nodes and unselected cells exactly, retires
selected-only nodes through explicit identity maps, replaces selected cells, and
checks the exact selected/outside cut. It does not allocate mass or qualify
physical geometry.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from human_skin_fat_retet_mapping import assemble_selected_retet

NODE_FIELDS=("points_donor_local_m_f64","parent_volume_node_row","full86_weights",
             "target_current_world_m","refinement_node_source_offsets",
             "refinement_node_source_rows","refinement_node_source_weights")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def readnpz(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}
def read_selection(path, base):
    with Path(path).open("r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        if reader.fieldnames is None or "source_tet_row" not in reader.fieldnames:
            raise ValueError("selection CSV must contain source_tet_row")
        rows=list(reader)
    if any(None in row for row in rows): raise ValueError("selection CSV has malformed row width")
    selected=[]
    for row in rows:
        i=int(row["source_tet_row"])
        if i<0 or i>=len(base["volume_tetrahedra"]): raise ValueError(f"selected source cell row {i} out of domain")
        for col,key in (("material_index","material_index"),("geometric_donor_body","geometric_donor_body"),("anatomical_region_id","anatomical_region_id")):
            if col in row and row[col] not in ("",None) and int(row[col])!=int(base[key][i]):
                raise ValueError(f"selection row {i} has stale/mismatched {col}")
        selected.append(i)
    if not selected: raise ValueError("selection CSV is empty")
    if len(set(selected))!=len(selected): raise ValueError("selection CSV repeats a source cell row")
    return np.asarray(selected,dtype=np.int64),rows

def compose_node_sidecars(base, retet, assembled):
    nbase=len(base["points_current_world_m_f32"])
    nout=len(assembled["points_current_world_m_f32"])
    output_map=np.asarray(assembled["retet_node_output_rows"],dtype=np.int64)
    local_base=np.asarray(retet["retet_node_base_rows"],dtype=np.int64)
    base_map=np.asarray(assembled["base_node_output_rows"],dtype=np.int64)
    new_local=np.flatnonzero(local_base<0)
    if base_map.shape!=(nbase,) or np.any(base_map < -1) or np.any(base_map>=nout):
        raise ValueError("assembled base-node map invalid")
    if len(np.unique(base_map[base_map>=0]))!=int(np.sum(base_map>=0)):
        raise ValueError("assembled base-node map is not injective")
    if output_map.shape!=(len(local_base),) or np.any(output_map<0) or np.any(output_map>=nout):
        raise ValueError("retet local-node output map invalid")
    for name in NODE_FIELDS:
        if name not in base: raise ValueError(f"base node sidecars missing {name}")
    side={"base_node_output_rows":base_map.copy(),
          "protected_base_node_rows":np.asarray(assembled["protected_base_node_rows"],dtype=np.int64).copy(),
          "protected_node_output_rows":np.asarray(assembled["protected_node_output_rows"],dtype=np.int64).copy(),
          "protected_node_local_rows":np.asarray(assembled["protected_node_local_rows"],dtype=np.int64).copy(),
          "protected_node_mapping_kind":np.asarray(assembled["protected_node_mapping_kind"]).copy()}
    scalar_names=("points_donor_local_m_f64","parent_volume_node_row","full86_weights","target_current_world_m")
    if len(new_local):
        missing=set(NODE_FIELDS)-set(retet)
        if missing: raise ValueError(f"new retet nodes lack source sidecars {sorted(missing)}")
        for name in NODE_FIELDS:
            values=np.asarray(retet[name])
            if name=="refinement_node_source_offsets":
                if values.shape!=(len(local_base)+1,): raise ValueError("retet source CSR offsets do not cover local nodes")
            elif name not in ("refinement_node_source_rows","refinement_node_source_weights") and values.shape[0]!=len(local_base):
                raise ValueError(f"retet sidecar {name} does not cover local nodes")
        for local,old in enumerate(local_base):
            if old<0: continue
            for name in scalar_names:
                if not np.array_equal(np.asarray(retet[name])[local],np.asarray(base[name])[int(old)]):
                    raise ValueError(f"retet changed retained sidecar {name} at base node {int(old)}")
    for name in scalar_names:
        old=np.asarray(base[name])
        combined=np.empty((nout,)+old.shape[1:],dtype=old.dtype)
        for oldrow,newrow in enumerate(base_map):
            if newrow>=0: combined[newrow]=old[oldrow]
        if len(new_local):
            vals=np.asarray(retet[name])
            for local in new_local: combined[output_map[local]]=vals[local]
        if len(np.unique(np.concatenate((base_map[base_map>=0],output_map[new_local]))))!=nout:
            raise ValueError(f"{name} does not map every output node exactly once")
        side[name]=combined
    old_off=np.asarray(base["refinement_node_source_offsets"],dtype=np.int64)
    old_rows=np.asarray(base["refinement_node_source_rows"],dtype=np.int64)
    old_weights=np.asarray(base["refinement_node_source_weights"],dtype=np.float64)
    row_lists=[None]*nout;weight_lists=[None]*nout
    for oldrow,newrow in enumerate(base_map):
        if newrow>=0:
            a,b=old_off[oldrow:oldrow+2]
            row_lists[newrow]=old_rows[a:b].copy();weight_lists[newrow]=old_weights[a:b].copy()
    if len(new_local):
        new_off=np.asarray(retet["refinement_node_source_offsets"],dtype=np.int64)
        new_rows=np.asarray(retet["refinement_node_source_rows"],dtype=np.int64)
        new_weights=np.asarray(retet["refinement_node_source_weights"],dtype=np.float64)
        if new_off[0]!=0 or new_off[-1]!=len(new_rows) or len(new_rows)!=len(new_weights) or np.any(np.diff(new_off)<0):
            raise ValueError("retet source CSR is malformed")
        for local in new_local:
            outrow=int(output_map[local]);a,b=new_off[local:local+2]
            row_lists[outrow]=new_rows[a:b].copy();weight_lists[outrow]=new_weights[a:b].copy()
    if any(x is None for x in row_lists): raise ValueError("node source CSR does not cover compact output")
    offsets=np.zeros(nout+1,dtype=np.int64)
    for i,x in enumerate(row_lists):offsets[i+1]=offsets[i]+len(x)
    side["refinement_node_source_offsets"]=offsets
    side["refinement_node_source_rows"]=np.concatenate(row_lists).astype(np.int64,copy=False)
    side["refinement_node_source_weights"]=np.concatenate(weight_lists).astype(np.float64,copy=False)
    return side


def jsonable(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    return x

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-npz",type=Path,required=True)
    ap.add_argument("--selected-cells-csv",type=Path,required=True)
    ap.add_argument("--retet-npz",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--receipt-json",type=Path,required=True)
    a=ap.parse_args()
    if a.output_dir.exists(): raise FileExistsError(f"output directory already exists: {a.output_dir}")
    if a.receipt_json.exists(): raise FileExistsError(f"receipt already exists: {a.receipt_json}")
    base=readnpz(a.base_npz);retet=readnpz(a.retet_npz)
    selected,selection_rows=read_selection(a.selected_cells_csv,base)
    candidate,report=assemble_selected_retet(base,selected,retet)
    sidecars=compose_node_sidecars(base,retet,candidate)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    geometry_path=a.output_dir/"geometry.npz"
    sidecar_path=a.output_dir/"node-sidecars.npz"
    np.savez_compressed(geometry_path,**candidate)
    np.savez_compressed(sidecar_path,**sidecars)
    selection_copy=a.output_dir/"selected-source-cells.csv"
    selection_copy.write_bytes(a.selected_cells_csv.read_bytes())
    report={"schema":"numi.selected-cell-retet-splice-671.v1","status":"SOURCE_PRESERVING_RETET_ASSEMBLED",
      "physical_admission":False,"mass_transfer":False,"native_or_gpu_run":False,
      "inputs":{"base_npz":{"path":str(a.base_npz),"sha256":sha(a.base_npz)},
        "selection_csv":{"path":str(a.selected_cells_csv),"sha256":sha(a.selected_cells_csv)},
        "retet_npz":{"path":str(a.retet_npz),"sha256":sha(a.retet_npz)}},
      "counts":report,
      "selection_rows":len(selection_rows),
      "retet_semantic_groups":report["semantic_groups_retet"],
      "source_registration":"not generated here; direct source330 child-cell registration is a separate packer input",
      "operational_reference":"not generated here; requires incident output child-tet support in the 54-column node map",
      "outputs":{"geometry_npz":{"path":str(geometry_path),"sha256":sha(geometry_path)},
        "node_sidecars_npz":{"path":str(sidecar_path),"sha256":sha(sidecar_path)},
        "selected_source_cells_csv":{"path":str(selection_copy),"sha256":sha(selection_copy)}}}
    a.receipt_json.parent.mkdir(parents=True,exist_ok=True)
    a.receipt_json.write_text(json.dumps(jsonable(report),indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps(jsonable(report),sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
