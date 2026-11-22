"""Compare exact crossing witnesses only after verifying both native packs' triangle indexing."""
import collections
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

def capture(run):
    j=json.loads((run/"launch.json").read_text())
    result=json.loads((run/"envelope/physical-banks.json").read_text())
    report=json.loads((run/"envelope/accepted-envelope.json").read_text())
    receipt=json.loads((run/"native/accepted-geometry"/f"step-{report['accepted_step']}.receipt.json").read_text())
    compressed=run/"terminal.mrvpack.zst"
    raw=subprocess.check_output(["/opt/homebrew/bin/zstd","-qdc",str(compressed)])
    h=hashlib.sha256(raw).hexdigest()
    if h!=receipt["pack_file_sha256"] or h!=report["pack_sha256"] or h!=result["pack_sha256"]:
        raise ValueError("Unbound terminal pack")
    sys.path.insert(0,str(Path(sys.argv[1]).parent/"src"))
    from numilab_human import common_atlas_skin_clearance as c
    head=c._PACK_HEADER.unpack_from(raw)
    directory={r[0]:r for i in range(head[2]) for r in [c._PACK_DIRECTORY.unpack_from(raw,c._PACK_HEADER.size+i*c._PACK_DIRECTORY.size)]}
    index,primitives=directory[3],directory[4]
    domain={}
    for i in range(primitives[4]):
        first,n,_,_,sem,_,_,stable=struct.unpack_from("<8I",raw,primitives[2]+64*i)
        if sem not in {51007,51005,51006,51011}:continue
        key=(sem,stable,i)
        offset=index[2]+4*first
        # Global index bases can differ. The full index stream must nevertheless agree.
        domain[key]=hashlib.sha256(memoryview(raw)[offset:offset+4*n]).hexdigest()
    witnesses=set()
    for line in (run/"envelope/target-witnesses.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row["semantic"] not in (51005,51006,51011):continue
        key=(row["semantic"],row["stable_id"],row["primitive"])
        if key not in domain or len(row["triangle_pairs"])!=row["count"]:raise ValueError("Incomplete witness set")
        witnesses.update((*key,s,t) for s,t in row["triangle_pairs"])
    if len(witnesses)!=result["physical_nonocular_pairs"]:raise ValueError("Unaccounted physical bank")
    a=j["argv"]; sources=[a[i] for i in (1,2,3)]+[a[a.index(flag)+1] for flag in ("--skin-payload","--soft-tissue-payload","--torso-anatomy-payload","--resting-anatomy-receipt")]
    return dict(hash=h,step=report["accepted_step"],sources=[j["pins"][p] for p in sources]),domain,witnesses

old,now,output=map(Path,sys.argv[2:]);a,ma,wa=capture(old);b,mb,wb=capture(now)
if ma!=mb or a["sources"]!=b["sources"]:raise ValueError("Different triangle topology or anatomical assets")
def by_type(w):return dict(collections.Counter(t[0] for t in w))
report=dict(prior=a,current=b,matched_surfaces=len(ma),prior_pairs=len(wa),current_pairs=len(wb),identical_pairs=len(wa&wb),
            new_pairs=len(wb-wa),cleared_pairs=len(wa-wb),persistent_by_semantic=by_type(wa&wb),new_by_semantic=by_type(wb-wa),
            retained_fraction_of_current=len(wa&wb)/len(wb),scope="two accepted endpoints at different physical times and bed placements, identical skin and internal triangle indexing")
output.write_text(json.dumps(report,indent=2)+"\n")
print(report["identical_pairs"],report["new_pairs"],report["cleared_pairs"])
