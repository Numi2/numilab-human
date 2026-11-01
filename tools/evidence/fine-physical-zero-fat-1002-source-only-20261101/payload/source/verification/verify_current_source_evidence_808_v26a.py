#!/usr/bin/env python3
"""808 recursive verifier for current source evidence with pinned 775/805/806 movement audits.

Direct current reports are validated first. When a step has no exact_context,
its hash-pinned parent is followed only if the exact D/I/O source-surface pins
remain unchanged. A malformed current report is fatal and is never skipped.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE / "verify_current_source_evidence_808_validation_v26a.py"
BASE_SHA256 = "6daa77251509436219b9ad4cb710ccaf4e4cf527a55358ad400c0b71e8881ef0"
MAX_ANCESTORS = 40

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def _load_base():
    if not BASE.is_file() or sha(BASE) != BASE_SHA256:
        raise ValueError("pinned 808 validation helper is missing or changed")
    spec=importlib.util.spec_from_file_location("current_source_evidence_base_808_v26a",BASE)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned 781 source-evidence validator")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

V=_load_base()

def _state_at(path):
    raw=V._load(path)
    state,_=V.V._unwrap_state(raw)
    if not isinstance(state,dict):
        raise ValueError("source verified-state is not an object")
    return state

def verify_current_source_evidence(state_path,state_override=None):
    """Verify direct current evidence or a hash-pinned unchanged-source chain."""
    current_path=Path(state_path).resolve()
    raw=V._load(current_path) if state_override is None else state_override
    state,_=V.V._unwrap_state(raw)
    if not isinstance(state,dict):
        raise ValueError("source verified-state is not an object")
    pins=V.V._source_pins(state)
    chain=[]
    seen=set()
    path=current_path
    current=state
    for depth in range(MAX_ANCESTORS):
        path=Path(path).resolve()
        if path in seen:
            raise ValueError("source evidence ancestry cycle")
        seen.add(path)
        V.V._check_not_747(path,"source evidence state")
        current_pins=V.V._source_pins(current)
        if current_pins!=pins:
            raise ValueError("source surface pins changed along unchanged-source ancestry")
        if "surface_crossings" in current and int(current["surface_crossings"])!=0:
            raise ValueError("source evidence state records nonzero crossings")
        if ("source_metadata_current_step_only" in current and
            current["source_metadata_current_step_only"] is not True):
            raise ValueError("source metadata is not current-step-only")
        chain.append({"path":str(path),"sha256":sha(path)})

        if "exact_context" in current and current.get("exact_context") is not None:
            if not isinstance(current["exact_context"],dict):
                raise ValueError("malformed current exact-context report reference")
            # Do not skip a current but invalid report in favor of any ancestor.
            anchor=V._validate_current_context(path,current,pins)
            if depth==0:
                return anchor
            return {
                "status":"PASS_INHERITED_CURRENT_SOURCE_EVIDENCE",
                "mode":"INHERITED_AUDITED_SOURCE_EVIDENCE_SAME_SOURCE_SURFACE_ANCESTRY",
                "state":{"path":str(current_path),"sha256":sha(current_path)},
                "source_surfaces":pins,
                "report":anchor["report"],
                "config":anchor["config"],
                "anchor_state":anchor["state"],
                "anchor_mode":anchor["mode"],
                "movement_by_role":anchor.get("movement_by_role",{}),
                "all185":anchor["all185"],
                "audit_implementation":anchor.get("audit_implementation"),
                "chain":chain,
                "physical_admission":False,
                "native_steps":0,
            }

        # source_state is the immediate predecessor. The inherited crossing
        # pointer may intentionally remain an older audit anchor and can have
        # different pins after an audited fit; it is not the ancestry link.
        link=current.get("source_state")
        if not isinstance(link,dict):
            link=current.get("source_crossing_evidence_inherited_from_unchanged_surfaces")
        if not isinstance(link,dict):
            raise ValueError("no current exact report or hash-pinned source ancestor")
        parent_path,_=V.V._verify_pin(link,label="unchanged-source parent state")
        V.V._check_not_747(parent_path,"unchanged-source parent state")
        parent_raw=V._load(parent_path)
        parent,_=V.V._unwrap_state(parent_raw)
        if not isinstance(parent,dict):
            raise ValueError("unchanged-source parent is not a verified-state object")
        # Check the parent pins immediately so a stale source ancestor cannot
        # be used to authorize a changed current candidate.
        if V.V._source_pins(parent)!=pins:
            raise ValueError("source surface pins changed along unchanged-source ancestry")
        current=parent
        path=parent_path
    raise ValueError("source evidence ancestry exceeds depth limit")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",required=True)
    a=ap.parse_args()
    print(json.dumps(verify_current_source_evidence(a.state),indent=2,sort_keys=True))

if __name__=="__main__":
    main()

