"""Audit the actual Float32 composition of the passing local fits."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

from envelope_lp import digest,exact_summary,prepare
from screen import banks,clearance,exact,hits,pair_index,skin,skin_faces


def changed_pair_surface(before,after,faces):
    delta=np.flatnonzero(np.any(before[faces]!=after[faces],axis=(1,2)))
    indices=[]
    for positions in (before,after):
        records=clearance._exact_surface_records(positions,faces)
        result=exact._audit_self_pairs_touching_faces(records,delta)
        indices.append(set(map(tuple,result["triangle_pairs"])))
    return {"changed_faces":len(delta),"before":len(indices[0]),"after":len(indices[1]),
            "introduced":len(indices[1]-indices[0]),
            "introduced_pairs":sorted(indices[1]-indices[0])}


def site_parity(points,faces,site_proof):
    locator=exact.prepare_point_location(clearance._exact_surface_records(points,faces))
    passed=[]
    for site in site_proof["exact_local_ray_parity"]:
        if site["location"]!="outside":
            continue
        p=exact.float32_point_lattice_key(site["world_float32_m"])
        result=exact.point_location_prepared_indexed(p,locator)
        passed.append({"route":site["route"],"site_order":site["site_order"],**result})
    return passed


if __name__=="__main__":
    launch,topo,bed,saved,summary_path,sites=map(Path,sys.argv[2:])
    base,tets,role,locked=prepare(launch,topo,bed)
    summary=json.loads(summary_path.read_text())
    assert digest(saved)==summary["proposal_sha256"]
    moved=base.copy()
    rows=np.load(saved,allow_pickle=False)
    moved[rows["node_rows"]]=rows["position_f32"]
    assert np.array_equal(base[locked],moved[locked])
    assert np.array_equal(base[:,1:],moved[:,1:])
    assert json.loads(json.dumps(exact_summary(moved,tets,role)))==summary["whole_volume"]
    exterior=moved[:len(skin)]
    route_proof=site_parity(exterior,skin_faces,json.loads(sites.read_text()))
    print(json.dumps({"exposed_site_status":route_proof},indent=2),file=sys.stderr,flush=True)
    tree=cKDTree(base.astype(float))
    layer_rows={}
    self_checks={"outer":changed_pair_surface(skin,exterior,skin_faces)}
    for name,key in (("inner",(51007,4294966545)),("deep",(51007,4294966546))):
        p,f=banks[key]
        distances,idx=tree.query(p)
        assert np.all(distances==0),name
        assert len(np.unique(idx))==len(idx),name
        changed=moved[idx]
        layer_rows[name]=(p,changed,f)
        self_checks[name]=changed_pair_surface(p,changed,f)
    # Analyze layers against each other only after moving their actual
    # physical rows. The unchanged external banks remain fixed as captured.
    layer_results=[]
    layers={"outer":(skin,exterior,skin_faces),**layer_rows}
    lname=list(layers)
    for i,name in enumerate(lname):
        x,y,f=layers[name]
        before_i=pair_index(x,f)
        after_i=pair_index(y,f)
        for other in lname[:i]:
            a,b,g=layers[other]
            t0,t1=hits(before_i,a,g),hits(after_i,b,g)
            layer_results.append({"layers":[name,other],"introduced":len(t1-t0),"before":len(t0),"after":len(t1)})
    tris=np.flatnonzero(np.any(skin[skin_faces]!=exterior[skin_faces],axis=(1,2)))
    before_i=pair_index(skin,skin_faces[tris])
    after_i=pair_index(exterior,skin_faces[tris])
    cross=[]
    for key,(p,f) in banks.items():
        if key[0]==51007:
            continue
        a,b=hits(before_i,p,f),hits(after_i,p,f)
        if a or b:
            cross.append({"bank":key,"before":len(a),"after":len(b),"introduced":len(b-a),
                          "introduced_capture_face_pairs":[(int(tris[i]),int(j)) for i,j in sorted(b-a)]})
    print(json.dumps({"outer_exposed_site_status":route_proof,"self":self_checks,
                      "layers":layer_results,"outer_changed_face_crossings":cross},indent=2))
