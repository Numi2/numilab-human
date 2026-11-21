"""Identify undefined contact responses directly from a captured post-dynamics GPU trial."""
import json, sys
from pathlib import Path
import numpy as np
from material_context import sha

def audit(assets, run):
    nodes = np.fromfile(run/"candidate-force-nodes.bin","<f4").reshape(-1,2,4)
    ports = np.fromfile(run/"candidate-force-ports.bin","<f4").reshape(512,4,4)
    lookup = np.fromfile(assets/"cortical-free-index.bin","<u4")
    cfg = json.loads((run/"launch.json").read_text())
    out = {"root":int(cfg["environment"]["NUMI_HUMAN_PELVIC_FORCE_WITNESS_ROOT"]),
           "node_sha256":sha(run/"candidate-force-nodes.bin"),
           "port_sha256":sha(run/"candidate-force-ports.bin"),"sides":[]}
    for side, block in enumerate(np.split(ports,2)):
        live = np.flatnonzero(block[:,1,0]>=0)
        witnesses = []
        for i in live:
            item = block[i]
            w = item[3]
            if not np.isfinite(w).all():
                continue
            ids = item[2,w!=0].astype(int)
            if np.any(ids<0) or np.any(ids>=len(lookup)):
                raise ValueError("Captured P1 ancestry outside material")
            absent = ids[lookup[ids]==np.iinfo(np.uint32).max]
            if len(absent):
                gap = float(item[1,1]);load=float(item[0,3])
                witnesses.append(dict(slot=int(side*256+i),pool=int(item[1,0]),missing_nodes=absent.tolist(),
                                      gap_m=gap if np.isfinite(gap) else str(gap),force_N=load if np.isfinite(load) else str(load)))
        node = np.split(nodes,2)[side]
        out["sides"].append(dict(side=side,slots=len(live),unmapped=witnesses,
            nonfinite_material_force_nodes=int((~np.isfinite(node[:,0,:3]).all(axis=1)).sum()),
            nonfinite_displacements=int((~np.isfinite(node[:,1,:3]).all(axis=1)).sum())))
    return out
if __name__=="__main__":
    assets,run,out=map(Path,sys.argv[1:])
    out.write_text(json.dumps(audit(assets,run),indent=2,allow_nan=False)+"\n")
