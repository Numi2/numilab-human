"""Reconstruct the existing fixed-cavity NPZ, without rerunning geometry fitting.
Requires the immutable accepted 003 base proposal. Native geometry, mass and
physical-step qualification remain separate; this tool only replays construction.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
BASE_SHA="a8e42f2da474a7297dea43c06414e8345f93f3a2ce63b72ccbf62bd76b61e22c"
def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()
def replay(base_path,report_path,output):
    if sha(base_path)!=BASE_SHA:raise ValueError("base cavity SHA mismatch")
    report=json.loads(Path(report_path).read_text())
    with np.load(base_path,allow_pickle=False) as base:
        offsets=base["replacement_tet_offsets"];parent_offsets=base["replacement_parent_offsets"]
        parent_rows=base["replacement_parent_cell_rows"];cells=base["replacement_tetrahedra"]
        source_sha=str(base["source_volume_sha256"].item())
        if source_sha!=report["source_sha256"]:raise ValueError("source volume identity mismatch")
        source_count=int(max(base["retained_input_cell_rows"].max(),parent_rows.max()))+1
        patches=report["connected_patches"];used={int(row) for patch in patches for row in patch["old_rows"]}
        old=[]
        for first,last in zip(offsets[:-1],offsets[1:]):
            first,last=int(first),int(last)
            rows=parent_rows[parent_offsets[first]:parent_offsets[first+1]].tolist()
            if set(rows)&used:
                if not set(rows)<=used:raise ValueError("incomplete base-cavity replacement")
            else:old.append({"old_rows":rows,"new_cells":cells[first:last].tolist()})
    patches=old+patches
    removed=np.asarray([row for patch in patches for row in patch["old_rows"]],np.int64)
    if len(np.unique(removed))!=len(removed):raise ValueError("overlapping source cavities")
    retained=np.flatnonzero(~np.isin(np.arange(source_count),removed))
    replacements=np.asarray([cell for patch in patches for cell in patch["new_cells"]],np.int64)
    offsets=np.r_[0,np.cumsum([len(patch["new_cells"]) for patch in patches])]
    parent_offsets=np.r_[0,np.cumsum([len(patch["old_rows"]) for patch in patches for _ in patch["new_cells"]])]
    parent_rows=np.asarray([row for patch in patches for _ in patch["new_cells"] for row in patch["old_rows"]],np.int64)
    if Path(output).exists():raise ValueError("output already exists")
    np.savez_compressed(output,retained_input_cell_rows=retained,replacement_tetrahedra=replacements,
        replacement_tet_offsets=offsets,replacement_parent_offsets=parent_offsets,
        replacement_parent_cell_rows=parent_rows,source_volume_sha256=np.array(source_sha))
    actual=sha(output)
    if actual!=report["proposal_sha256"]:raise ValueError("replayed proposal SHA mismatch: "+actual)
    return {"status":"EXACT_CONSTRUCTION_REPLAY","sha256":actual,"cavities":len(patches),
            "mass_qualified":False,"native_steps":0}
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cavities",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(replay(args.base_cavities,args.report,args.output),sort_keys=True))
