"""Compare three skin boundaries to the same opening skeleton and native anatomy banks."""
import json
import runpy
import subprocess
import sys
from pathlib import Path
import numpy as np
from run_bed_work import digest
from verify_native_extensor_pair import loaded


def compare(case,paths,reader,banks,out):
    r,poses,raw=loaded(case,True,False)
    provenance=json.loads((paths/"connected-columns.json").read_text())
    boundary=paths/"outer-boundaries.npz"
    if (provenance["datum_pose"]!=digest(case/"terminal.receipt.json") or
        provenance["outer_boundaries"]!=digest(boundary) or not provenance["registration"]["single_rigid_transform"] or
        not provenance["loaded_state"]["single_rigid_transform"]):
        raise ValueError("Surface or opening pose mismatch")
    sys.path.insert(0,str(reader.parent/"src"))
    api=runpy.run_path(str(reader));scan=runpy.run_path(str(banks))["scan_banks"]
    ((_,_,skin,faces),)=api["primitives"](raw,poses,skin_only=True)
    with np.load(boundary) as p:
        triangles=np.ascontiguousarray(p["triangles"],dtype="i4")
        surfaces=[("native_opening",skin,faces),("dense_reference",p["reference_xyz"],triangles),("dense_loaded",p["loaded_xyz"],triangles)]
    out.mkdir()
    for name,v,f in surfaces:
        directory=out/name;directory.mkdir()
        proof=dict(pack_sha256=r["base_pack_file_sha256"],opening_sha256=provenance["datum_pose"],
                   boundary_sha256=provenance["outer_boundaries"],skin_vertices=len(v),skin_faces=len(f),
                   scope=f"Matched native opening targets with {name} skin. Material boundaries are static observations.")
        scan(api,raw,poses,np.ascontiguousarray(v,dtype="f4"),np.ascontiguousarray(f,dtype="i4"),proof,directory,"geometry.json",check_skin=True)
        subprocess.run([sys.executable,str(banks.parent/"classify_registered_envelope.py"),str(directory/"geometry.json"),str(case/"stdout.log"),
                        str(directory/"physical-banks.json"),"--window-receipt",str(case/"resting-ocular-sclera-window.json")],check=True)
if __name__=="__main__":compare(*(Path(s).resolve() for s in sys.argv[1:]))
