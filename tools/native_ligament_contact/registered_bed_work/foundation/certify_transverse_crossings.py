"""Classify the known intersecting pairs using exact binary32 plane-side predicates."""
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

from run_bed_work import digest


def certify(case, reader, opening, evidence, initial):
    sys.path.insert(0,str(reader.parent/"src"))
    api=runpy.run_path(str(reader))
    exact=api["intersections"]
    def read(file):
        return json.loads(file.read_text())
    job=read(case/"launch.json")
    receipt=read(case/"terminal.receipt.json")
    registration=read(opening/"terminal.receipt.json")["initial_anatomical_registration"]
    sha=receipt["base_pack_file_sha256" if initial else "pack_file_sha256"]
    classified=read(evidence/"physical-banks.json")
    if sha != classified["pack_sha256"]:
        raise ValueError("Different witness pack")
    raw=subprocess.check_output(["zstd","-qdc",case/("initial.mrvpack.zst" if initial else "accepted.mrvpack.zst")])
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError("Geometry digest mismatch")
    poses=registration["body_poses"] if initial else receipt["accepted_body_poses"]
    poses={b["body_index"]:b for b in poses}
    tissue=Path(job["argv"][job["argv"].index("--soft-tissue-payload")+1])
    manifests=read(tissue.with_suffix(".manifest.json"))
    if digest(tissue)!=manifests["payload"]["sha256"] or digest(tissue)!=job["pins"][str(tissue)]:
        raise ValueError("Muscle identity drift")
    names={r["stable_id"]:r.get("label",r.get("source_name")) for r in manifests["source"]["surfaces"]}
    banks={b["primitive"]:b for b in map(json.loads,(evidence/"target-witnesses.jsonl").open())
           if not b["ocular_source_id"] and not (b["semantic"]==51010 and b["stable_id"]>=4294966700)}
    (skin,)=api["primitives"](raw,poses,skin_only=True)
    def triangles(xyz,faces):
        cache={}
        def get(i):
            if i not in cache:
                x=tuple(exact.float32_point_lattice_key(v) for v in xyz[faces[i]])
                n=exact._cross(exact._sub(x[1],x[0]),exact._sub(x[2],x[0]))
                cache[i]=(x,n)
            return cache[i]
        return get
    def straddles(v,p):
        d=[exact._dot(p[1],exact._sub(x,p[0][0])) for x in v[0]]
        return min(d)<0<max(d)
    surface=triangles(*skin[2:]);result=[]
    for key,idx,xyz,faces in api["primitives"](raw,poses):
        if idx not in banks:continue
        b=banks.pop(idx)
        if key!=(b["semantic"],b["stable_id"]):raise ValueError("Primitive and anatomical IDs differ")
        target=triangles(xyz,faces);hits=b["triangle_pairs"];proper=0;first=None
        for i,j in hits:
            a,b=surface(i),target(j)
            if straddles(a,b) and straddles(b,a):
                proper+=1
                if first is None:first=[i,j]
        if hits:result.append(dict(semantic=key[0],stable_id=key[1],primitive=idx,
                         muscle=names.get(key[1]) if key[0]==51005 else None,
                         pairs=len(hits),strict_transverse_pairs=proper,witness=first))
    if banks or sum(r["pairs"] for r in result)!=classified["physical_nonocular_pairs"]:
        raise ValueError("Witness or ownership accounting differs")
    result.sort(key=lambda r:-r["strict_transverse_pairs"])
    return dict(native_pack=sha,physical_pairs=classified["physical_nonocular_pairs"],
                strict_transverse_pairs=sum(r["strict_transverse_pairs"] for r in result),banks=result)


if __name__=="__main__":
    case,reader,opening,baseline,out=(Path(a).resolve() for a in sys.argv[1:])
    results={"opening":certify(case,reader,opening,baseline,True),
             "accepted":certify(case,reader,opening,case/"envelope",False)}
    out.write_text(json.dumps(results,indent=2)+"\n")
