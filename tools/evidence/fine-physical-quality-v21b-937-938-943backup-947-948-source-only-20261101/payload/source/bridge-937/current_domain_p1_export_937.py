#!/usr/bin/env python3
"""Versioned current-domain pin adapter for the unchanged 217 P1 exporter.

All row-count and file-hash gates remain active. The only substitutions are
explicit pins supplied by a hash-pinned current-domain configuration; the
source007 owner and label-exporter files themselves must match frozen SHAs.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, tempfile
from pathlib import Path
from typing import Any
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
P=H.parent
P1_SOURCE=H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/p1_reference_anchor_217.py"
P1_SOURCE_SHA="67d6a0f3dd4f345433305e154e906f73b4f24633d9f3805103f496ce164846f8"
LABEL_SOURCE=H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/export_child_semantic_labels_217.py"
LABEL_SOURCE_SHA="e2075c47e867dc56629d066db8b4409c557b0b8bf1069ba192aa2108355c7910"
CONFIG_SCHEMA="numi.native-domain-217.current-p1-owner-pins.v1"
FULL86_SHA="98205ae7de44401544b7969184788f5c2ee39cd4404cd34b8037377e7ae63a76"
OLD_FAT_VOLUME_SHA="7ab13a433e0cb08350c901388c0c86712269e0cf66ce70df5fe33de248357e76"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def require_pin(value,label):
    if not isinstance(value,dict) or not isinstance(value.get("path"),str) or not isinstance(value.get("sha256"),str):
        raise ValueError(f"{label} must contain path and sha256")
    p=Path(value["path"]).resolve()
    if not p.is_file() or sha(p)!=value["sha256"]:raise ValueError(f"{label} file pin mismatch")
    return {"path":str(p),"sha256":value["sha256"]}

def _load_p1_owner(config:dict):
    if sha(P1_SOURCE)!=P1_SOURCE_SHA:raise ValueError("pinned source007 P1 owner SHA changed")
    # One explicit versioned generalization: the old Fat support volume hash is
    # made a module constant. It is set from a required current-volume pin below;
    # all other original structural/hash/status checks remain unchanged.
    src=P1_SOURCE.read_text()
    old='"'+OLD_FAT_VOLUME_SHA+'"'
    if src.count(old)!=1:raise ValueError("cannot uniquely identify frozen source007 Fat volume literal")
    src=src.replace(old,"FAT_SUPPORT_VOLUME_SHA256")
    if sha(LABEL_SOURCE)!=LABEL_SOURCE_SHA:raise ValueError("pinned source007 child-label exporter SHA changed")
    roles=config["roles"]; counts=config["counts"]; combined=config["combined"]; source=config["source"]
    mod=importlib.util.module_from_spec(importlib.util.spec_from_loader("p1_reference_anchor_937_current",loader=None))
    mod.__file__=str(P1_SOURCE); mod.__package__=""
    mod.FAT_SUPPORT_VOLUME_SHA256=roles["fat"]["current_volume"]["sha256"]
    exec(compile(src,str(P1_SOURCE),"exec"),mod.__dict__)
    mod.SKIN_CELLS=int(counts["skin_cells"])
    mod.FAT_CELLS=int(counts["fat_cells"])
    mod.COMBINED_CELLS=int(counts["combined_cells"])
    mod.P1_PINS={}
    for key,role in (("skin","skin"),("fat","fat")):
        q=roles[role]
        mod.P1_PINS[key]={"support_sha256":q["support_npz"]["sha256"],
          "receipt_sha256":q["support_receipt"]["sha256"],"cells":int(q["cell_count"]),
          "material":int(q["material_index"]),"status":q["status"]}
    map_names={"element_map":"ELEMENT_MAP_SHA256","handoff_receipt":"ADAPTER_RECEIPT_SHA256",
      "combined_node":"COMBINED_NODE_SHA256","combined_ele":"COMBINED_ELE_SHA256",
      "combined_node_map":"COMBINED_NODE_MAP_SHA256","skin_node":"SKIN_NODE_SHA256",
      "fat_node":"FAT_NODE_SHA256","skin_ele":"SKIN_ELE_SHA256","fat_ele":"FAT_ELE_SHA256"}
    for name,global_name in map_names.items():setattr(mod,global_name,combined[name]["sha256"])
    # Full86 stays bound to the immutable source330 weight table owner constant.
    if source["full86_weights"]["sha256"]!=FULL86_SHA:raise ValueError("current config attempts to replace immutable source330 full86 weights")
    mod.FULL86_SHA256=FULL86_SHA
    mod.DEFAULT_PROJECTION_HELPER_PATH=Path(source["projection_helper"]["path"])
    mod.DEFAULT_PROJECTION_HELPER_SHA256=source["projection_helper"]["sha256"]
    return mod

def read_current_state(path:Path)->dict:
    raw=json.loads(Path(path).read_text())
    queue=[raw];seen=set()
    while queue:
        q=queue.pop(0)
        if not isinstance(q,dict) or id(q) in seen:continue
        seen.add(id(q))
        if isinstance(q.get("Skin_current"),dict) and isinstance(q.get("Fat_current"),dict):
            if q.get("physical_admission") is True or q.get("native_steps",0)!=0:
                raise ValueError("current P1 state is not explicitly source-only")
            if q.get("mass_mapping",q.get("mass_reconciliation","PENDING")) not in ("PENDING","NOT_RUN","PENDING_VOLUME_CHANGE_RECONCILIATION"):
                raise ValueError("current P1 state has unexpected mass status")
            return q
        for key in ("current_verified_state_20261031","current","verified_state","state"):
            if isinstance(q.get(key),dict):queue.append(q[key])
    raise ValueError("current state pin lacks paired role volume entries")


def _validate_config(config_path:Path,expected_sha:str)->tuple[dict,dict]:
    config_path=config_path.resolve()
    if sha(config_path)!=expected_sha:raise ValueError("current-domain pin config SHA mismatch")
    c=json.loads(config_path.read_text())
    if c.get("schema")!=CONFIG_SCHEMA:raise ValueError("unsupported current-domain P1 pin schema")
    for name,want in (("p1_owner",{"path":P1_SOURCE,"sha256":P1_SOURCE_SHA}),
                      ("label_exporter",{"path":LABEL_SOURCE,"sha256":LABEL_SOURCE_SHA})):
        got=require_pin(c.get("owners",{}).get(name),name)
        if Path(got["path"])!=want["path"].resolve() or got["sha256"]!=want["sha256"]:
            raise ValueError(f"{name} is not the exact frozen current owner")
    current_state=require_pin(c.get("current_state"),"current_state")
    state_core=read_current_state(Path(current_state["path"]))
    counts=c.get("counts",{})
    if any(not isinstance(counts.get(k),int) or counts[k]<=0 for k in ("skin_cells","fat_cells")):
        raise ValueError("current role cell counts must be positive integers")
    if counts.get("combined_cells")!=counts["skin_cells"]+counts["fat_cells"]:
        raise ValueError("combined count does not equal current Skin+Fat counts")
    if set(c.get("roles",{}))!={"skin","fat"}:raise ValueError("current support config must contain exactly Skin and Fat")
    for role,mat in (("skin",15),("fat",16)):
        q=c["roles"][role]
        if q.get("cell_count")!=counts[role+"_cells"] or q.get("material_index")!=mat:
            raise ValueError(f"{role} current row count/material pin mismatch")
        if not isinstance(q.get("status"),str) or not q["status"].startswith("PASS_SOURCE_ONLY_"):
            raise ValueError(f"{role} support is not source-only PASS")
        for key in ("support_npz","support_receipt","current_volume","node","ele"):
            q[key]=require_pin(q.get(key),f"{role}.{key}")
        receipt=json.loads(Path(q["support_receipt"]["path"]).read_text())
        if receipt.get("status")!=q["status"] or receipt.get("physical_admission") is not False or receipt.get("native_steps",0)!=0:
            raise ValueError(f"{role} support receipt does not retain source-only status")
        role_state=state_core[role.capitalize()+"_current"]
        with np.load(q["current_volume"]["path"],allow_pickle=False) as z:
            if "tetrahedra" not in z or len(z["tetrahedra"])!=q["cell_count"]:
                raise ValueError(f"{role} current volume cell row count differs from pin")
            if "canonical_parent_union_offsets" not in z or len(z["canonical_parent_union_offsets"])!=q["cell_count"]+1:
                raise ValueError(f"{role} volume lacks full canonical ancestry CSR")
            if "canonical_parent_union_current_state_path" not in z or "canonical_parent_union_current_state_sha256" not in z:
                raise ValueError(f"{role} volume lacks the current-state ancestry binding")
            if (Path(str(np.asarray(z["canonical_parent_union_current_state_path"]).item())).resolve()!=Path(current_state["path"])
                    or str(np.asarray(z["canonical_parent_union_current_state_sha256"]).item())!=current_state["sha256"]):
                raise ValueError(f"{role} volume current-state ancestry differs from explicit config pin")
            if "canonical_parent_union_current_volume_path" not in z or "canonical_parent_union_current_volume_sha256" not in z:
                raise ValueError(f"{role} volume lacks the input current-volume ancestry binding")
            if (Path(str(np.asarray(z["canonical_parent_union_current_volume_path"]).item())).resolve()!=Path(role_state.get("path","")).resolve()
                    or str(np.asarray(z["canonical_parent_union_current_volume_sha256"]).item())!=role_state.get("sha256")):
                raise ValueError(f"{role} volume ancestry differs from the paired current-state input")
            if "material_indices" not in z or not np.all(np.asarray(z["material_indices"])==mat):
                raise ValueError(f"{role} current volume material IDs differ from role contract")
    combined=c.get("combined",{})
    for key in ("element_map","handoff_receipt","combined_node","combined_ele","combined_node_map"):
        combined[key]=require_pin(combined.get(key),f"combined.{key}")
    for role in ("skin","fat"):
        for key in ("node","ele"): combined[f"{role}_{key}"]=c["roles"][role][key]
    handoff=json.loads(Path(combined["handoff_receipt"]["path"]).read_text())
    if handoff.get("physical_admission") is not False or handoff.get("native_steps")!=0 or handoff.get("mass_mapping")!="NOT_RUN":
        raise ValueError("combined 217 handoff is not source-only / mass-pending")
    inputs=handoff.get("inputs",{})
    if inputs.get("current_state",{}).get("sha256")!=current_state["sha256"] or Path(inputs.get("current_state",{}).get("path","")).resolve()!=Path(current_state["path"]):
        raise ValueError("combined 217 handoff is bound to another current state")
    for role in ("skin","fat"):
        key=role+"_volume"
        want=c["roles"][role]["current_volume"]
        got=inputs.get(key,{})
        if Path(got.get("path","")).resolve()!=Path(want["path"]) or got.get("sha256")!=want["sha256"]:
            raise ValueError(f"combined 217 handoff {role} volume differs from explicit current pin")
    source=c.get("source",{})
    for key in ("parent_mass_csv","full86_weights","projection_helper"):
        source[key]=require_pin(source.get(key),f"source.{key}")
    if source["full86_weights"]["sha256"]!=FULL86_SHA:raise ValueError("source full86 weights hash changed")
    # Only existing pinned projection helper is authorized by this owner version.
    if source["projection_helper"]["sha256"]!="02b04cff325b4adea24d055fdfd2121ab4d596a3c075ab6fe8183094eb46e1d6":
        raise ValueError("projection helper is not the pinned source007 helper")
    return c,{"config":pin_obj(config_path),"current_state":current_state,"counts":counts,"roles":c["roles"],"combined":combined,"source":source}

def pin_obj(path:Path)->dict:
    return {"path":str(path.resolve()),"sha256":sha(path)}

def _parse_flags(argv:list[str])->dict[str,str|bool]:
    out={};i=0
    while i<len(argv):
        x=argv[i]
        if x=="--":i+=1;continue
        if x.startswith("--"):
            if "=" in x:
                k,v=x.split("=",1);out[k]=v;i+=1
            elif i+1<len(argv) and not argv[i+1].startswith("--"):
                out[x]=argv[i+1];i+=2
            else:out[x]=True;i+=1
        else:i+=1
    return out

def _assert_cli_matches(argv:list[str],c:dict,validated:dict)->dict[str,Any]:
    f=_parse_flags(argv)
    required=["--p1-supported-reference-anchor","--p1-skin-support-npz","--p1-skin-support-receipt-json",
      "--p1-fat-support-npz","--p1-fat-support-receipt-json","--combined-element-map-csv-gz",
      "--combined-node-map-csv-gz","--combined-handoff-receipt-json","--child-node","--child-ele",
      "--parent-mass-csv","--parent-full86-weights-csv","--projection-helper-path","--projection-helper-sha256"]
    missing=[x for x in required if x not in f]
    if missing:raise ValueError("P1 current mode missing required args: "+", ".join(missing))
    if f.get("--p1-supported-reference-anchor") is not True:raise ValueError("must use authenticated P1 anchor mode")
    if "--same-material-current-reference-registration" in f:raise ValueError("global centroid registration cannot replace current P1 anchor mode")
    if f.get("--capacity-profile")!="full-reference-500k-nodes-2m-cells":raise ValueError("current full-domain capacity profile must be explicit")
    expected={"--p1-skin-support-npz":validated["roles"]["skin"]["support_npz"]["path"],
      "--p1-skin-support-receipt-json":validated["roles"]["skin"]["support_receipt"]["path"],
      "--p1-fat-support-npz":validated["roles"]["fat"]["support_npz"]["path"],
      "--p1-fat-support-receipt-json":validated["roles"]["fat"]["support_receipt"]["path"],
      "--combined-element-map-csv-gz":validated["combined"]["element_map"]["path"],
      "--combined-node-map-csv-gz":validated["combined"]["combined_node_map"]["path"],
      "--combined-handoff-receipt-json":validated["combined"]["handoff_receipt"]["path"],
      "--child-node":validated["combined"]["combined_node"]["path"],
      "--child-ele":validated["combined"]["combined_ele"]["path"],
      "--parent-mass-csv":validated["source"]["parent_mass_csv"]["path"],
      "--parent-full86-weights-csv":validated["source"]["full86_weights"]["path"],
      "--projection-helper-path":validated["source"]["projection_helper"]["path"],
      "--projection-helper-sha256":validated["source"]["projection_helper"]["sha256"]}
    for flag,want in expected.items():
        if str(Path(str(f[flag])).resolve())!=want if flag not in ("--projection-helper-sha256",) else str(f[flag])!=want:
            raise ValueError(f"{flag} differs from explicit current-domain pins")
    if "--diagnostic-child-prefix-count" in f:raise ValueError("prefix-only P1 export is not accepted by current-domain wrapper")
    return f

def run(current_pins:Path,current_pins_sha256:str,argv:list[str])->int:
    c,v=_validate_config(current_pins,current_pins_sha256)
    flags=_assert_cli_matches(argv,c,v)
    owner=_load_p1_owner(c)
    sys.path.insert(0,str(LABEL_SOURCE.parent))
    spec=importlib.util.spec_from_file_location("export_child_semantic_labels_937_current",LABEL_SOURCE)
    if spec is None or spec.loader is None:raise ImportError(str(LABEL_SOURCE))
    exporter=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=exporter;spec.loader.exec_module(exporter)
    exporter.p1_anchor=owner
    old=list(sys.argv)
    try:
        sys.argv=[str(LABEL_SOURCE)]+argv
        rc=int(exporter.main())
    finally:sys.argv=old
    if rc==0:
        label_path=Path(str(flags["--output-labels-csv"])).resolve()
        receipt_path=Path(str(flags.get("--receipt-json",str(label_path)+".receipt.json"))).resolve()
        if not receipt_path.is_file():raise ValueError("source007 P1 exporter did not write its full receipt")
        doc=json.loads(receipt_path.read_text())
        doc["current_domain_pins"]={"schema":CONFIG_SCHEMA,"config":pin_obj(current_pins),
          "p1_owner":{"path":str(P1_SOURCE),"sha256":P1_SOURCE_SHA},
          "label_exporter":{"path":str(LABEL_SOURCE),"sha256":LABEL_SOURCE_SHA},
          "current_state":v["current_state"],"counts":v["counts"],"roles":{k:{"current_volume":v["roles"][k]["current_volume"],
            "support_npz":v["roles"][k]["support_npz"],"support_receipt":v["roles"][k]["support_receipt"],
            "cell_count":v["roles"][k]["cell_count"],"material_index":v["roles"][k]["material_index"],
            "status":v["roles"][k]["status"]} for k in ("skin","fat")},
          "combined":v["combined"],"source":v["source"],
          "legacy_static_hashes_replaced_by_explicit_pins":True,
          "original_domain_structure_and_semantic_gates_retained":True,
          "physical_admission":False,"native_steps":0,"mass_transport":"NOT_RUN"}
        tmp=receipt_path.with_name(receipt_path.name+".current-pin.tmp")
        tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n");os.replace(tmp,receipt_path)
    return rc

def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--current-domain-pins",type=Path,required=True)
    ap.add_argument("--current-domain-pins-sha256",required=True)
    ap.add_argument("owner_args",nargs=argparse.REMAINDER)
    a=ap.parse_args()
    argv=a.owner_args[1:] if a.owner_args[:1]==["--"] else a.owner_args
    return run(a.current_domain_pins,a.current_domain_pins_sha256,argv)
if __name__=="__main__":raise SystemExit(main())
