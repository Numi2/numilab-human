"""Attribute exact skin folds to the native hand and upper-leg chains."""
import json
from pathlib import Path
import struct

import numpy as np

from prepare_skin import HERE, bindings, capture, decode

def main():
    prior=HERE.parent/"resting-clearance-and-atrium-002"
    p=prior/"rebound-original-16"
    deck=json.loads((p/"launch.json").read_text())
    rigid=Path(deck["argv"][1])
    assert bindings.audit.sha(rigid)==deck["pins"][str(rigid)]
    raw=rigid.read_bytes()
    header=struct.unpack_from("<8s10I32s",raw)
    assert header[0]==b"NHRIGID2" and header[1]==1
    bodies,joints=header[4:6]
    rows=[struct.unpack_from("<2I",raw,224+160*bodies+144*i) for i in range(joints)]
    def subtree(root):
        owned={root}
        while True:
            grown=owned | {child for parent,child in rows if parent in owned}
            if grown==owned:
                return owned
            owned=grown
    # Body 45/95 is the distal wrist chain, 128/143 is the pelvis side.
    groups={"left_hand":subtree(45), "right_hand":subtree(95),
            "left_hip_and_leg":subtree(128),"right_hip_and_leg":subtree(143)}
    results={}
    options=[(p,prior/"rebound-original-16-exact.json",prior/"skin-rebound.nhskin"),
             (HERE/"wider-release-control-16",HERE/"wider-release-control-16-exact.json",prior/"skin-rebound.nhskin"),
             (HERE/"late-rebind-control-16",HERE/"late-rebind-control-16-exact.json",HERE/"skin-rebound-late.nhskin")]
    for run, audit, skin in options:
        if not audit.is_file():
            continue
        exact=json.loads(audit.read_text())
        receipt,_,faces=capture(run/"native/accepted-geometry/step-8000.mrvpack")
        assert receipt["pack_file_sha256"]==exact["accepted_pack_sha256"]
        model=decode(skin.read_bytes())
        owner=model["binds"]["body"][np.argmax(model["weights"],axis=1)]
        pair=faces[exact["self_intersections"]["triangle_pairs"]]
        index=owner[pair]
        typed={k:np.isin(index,list(v)).any(axis=2) for k,v in groups.items()}
        result={}
        for side in ("left","right"):
            a,b=[typed[f"{side}_{region}"] for region in ("hand","hip_and_leg")]
            result[f"{side}_hand_to_hip_or_leg"]=int(((a[:,0]&b[:,1])|(a[:,1]&b[:,0])).sum())
        result["skin_self_pairs"]=len(pair)
        result["pack_sha256"]=receipt["pack_file_sha256"]
        results[run.name]=result
    (HERE/"skin-contact-owners.json").write_text(json.dumps(
        {"ownership":"at least one dominant source influence on each separate grouped triangle",
         "chains":{k:sorted(v) for k,v in groups.items()},"results":results},indent=2)+"\n")
    print(results)
if __name__=="__main__":
    main()
