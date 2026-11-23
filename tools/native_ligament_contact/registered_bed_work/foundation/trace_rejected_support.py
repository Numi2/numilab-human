"""Measure registered material lengths on the native line of load at the first failing source point."""
import csv
import hashlib
import json
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from inspect_bed_load_paths import intersections
from run_bed_work import digest


def trace(case, source, reader, opening):
    sys.path.insert(0,str(reader.parent/"src"))
    api=runpy.run_path(str(reader))
    receipt=json.loads((case/"terminal.receipt.json").read_text())
    cfg=json.loads((case/"launch.json").read_text())
    fault=json.loads(source.read_text()); v=fault["vertex"]
    if fault["material_step_index"] != receipt["accepted_step"]:
        raise ValueError("Expected the last valid pose before the failing trial")
    meta=receipt["skin_source_mapping"]
    with (case/"registered-bed-columns.csv").open() as f:
        column,=(r for r in csv.DictReader(f) if int(r["vertex"])==v)
    payload=Path(cfg["argv"][cfg["argv"].index("--soft-tissue-payload")+1])
    m=json.loads(payload.with_suffix(".manifest.json").read_text())
    if cfg["pins"][str(payload)]!=digest(payload) or cfg["pins"][str(payload)]!=m["payload"]["sha256"]:
        raise ValueError("Inconsistent native muscle mesh")
    names={r["stable_id"]:r.get("label",r.get("source_name")) for r in m["source"]["surfaces"]}
    frames={}
    for name in ("initial","accepted"):
        raw=subprocess.check_output(["zstd","-qdc",case/(name+".mrvpack.zst")])
        field="base_pack_file_sha256" if name=="initial" else "pack_file_sha256"
        if hashlib.sha256(raw).hexdigest()!=receipt[field]:raise ValueError("Wrong captured geometry")
        h,d=api["clearance"]._PACK_HEADER,api["clearance"]._PACK_DIRECTORY
        parts={r[0]:r for r in (d.unpack_from(raw,h.size+j*d.size) for j in range(h.unpack_from(raw)[2]))}
        vertex,idx,prim,inst=(parts[i] for i in range(2,6))
        glob=meta["pack_first_vertex"]+v
        for j in range(prim[4]):
            off,n,_,which,sem,_,_,ident=struct.unpack_from("<8I",raw,prim[2]+j*prim[5])
            if (sem,ident)!=api["SKIN"]:continue
            triangles=np.frombuffer(raw,dtype="<u4",count=n,offset=idx[2]+4*off)
            if glob not in triangles:raise ValueError("Source witness is unreferenced")
            transform=struct.unpack_from("<8f12I",raw,inst[2]+which*inst[5])
            if transform[10]!=0:raise ValueError("Unknown skin instance binding")
            position=struct.unpack_from("<3f",raw,vertex[2]+glob*vertex[5])
            point=api["world_positions"](np.array([position],dtype="f4"),transform)[0].astype(float)
            break
        else:raise ValueError("Missing skin")
        data=json.loads((opening/"terminal.receipt.json").read_text())["initial_anatomical_registration"]["body_poses"] if name=="initial" else receipt["accepted_body_poses"]
        poses={b["body_index"]:b for b in data}
        surfaces=[]
        for key,p,xyz,faces in api["primitives"](raw,poses):
            if key[0] not in (51004,51005):continue
            hits=intersections(xyz,faces,point)
            if hits and any(0 < z <= .08 for z,i in hits):
                surfaces.append(dict(semantic=key[0],stable_id=key[1],primitive=p,muscle=names.get(key[1]) if key[0]==51005 else None,
                                     heights_m=[z for z,i in hits],faces=[i for z,i in hits]))
        surfaces.sort(key=lambda b:min(z for z in b["heights_m"] if z>0))
        if name=="initial" and np.linalg.norm(point-np.array([float(column[k]) for k in ("x_m","y_m","z_m")]))>2e-6:
            raise ValueError("Released point differs from its source footprint")
        first_bone=next(min(z for z in b["heights_m"] if z>0) for b in surfaces if b["semantic"]==51004)
        first_muscle=next(b for b in surfaces if b["semantic"]==51005)
        frames[name]=dict(point_m=point.tolist(),first_bone_boundary_m=first_bone,first_muscle=first_muscle["muscle"],
                          first_muscle_boundary_m=min(z for z in first_muscle["heights_m"] if z>0),surfaces=surfaces)
    return dict(accepted_step=receipt["accepted_step"],skin_vertex=v,thickness_source_vertex=int(column["thickness_source_vertex"]),
                nominal_dermis_m=float(column["skin_m"]),nominal_fat_m=float(column["fat_m"]),fault=fault,
                tissue_source_sha256=digest(payload),accepted_pack_sha256=receipt["pack_file_sha256"],frames=frames,
                scope="Vertical intersections of the bed-normal ray with the registered muscles and bones. Distances measure geometry only. Contact connectivity, material identity in uncovered intervals and load distribution require a coupled mechanical model.")


if __name__=="__main__":
    case,source,reader,opening,out=(Path(s).resolve() for s in sys.argv[1:])
    out.write_text(json.dumps(trace(case,source,reader,opening),indent=2,allow_nan=False)+"\n")
