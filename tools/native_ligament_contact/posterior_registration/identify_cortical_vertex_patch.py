"""Locate the measured switching contacts in the registered partition's original cortical facets."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from qualify_material_motion import frame
from material_context import sha
def inspect(part,assets,poses,features,sites,out):
    import csv
    d=json.loads(features.read_text());values={z['site'] for k,row in d['witnesses'].items() if k.startswith('1/') for z in row['rows'][:2]}
    reg={r['body_index']:frame(r) for r in json.loads(poses.read_text())['initial_anatomical_registration']['body_poses']}
    surface=np.load(part/'conforming-boundary/30.npz');p=np.load(part/'projected/30.npz');ports=np.load(part/'ports/30.npz')
    pool=np.fromfile(assets/'pool-material-points.bin','f4').reshape(-1,7,4)
    with sites.open() as f:rr=list(csv.DictReader(f))
    mesh=surface['vertices'];tri=surface['face'];source=surface['source_face'];records={}
    for site in sorted(values):
        bound=pool[site];body=int(bound[3].copy().view('u4')[2]);q=reg[body];y=Rotation.from_quat(q[3:]).apply(bound[2,:3])+q[:3]
        distance=np.linalg.norm(mesh-y,axis=1);i=int(distance.argmin());incident=np.flatnonzero((tri==i).any(1))
        tags={};refs=reg[128];rot=Rotation.from_quat(refs[3:]).inv()
        for k in incident:
            s=source[k]
            if s>=len(p['port_vertices']):tag='external_muscle';face=-1
            else:tag=str(ports['target_bank'][s]);face=int(ports['target_face'][s])
            a,b,c=mesh[tri[k]];n=np.cross(b-a,c-a);n=rot.apply(n/np.linalg.norm(n))
            tags.setdefault((tag,face),[]).append(-n)
        normals={k:np.mean(ns,axis=0) for k,ns in tags.items()};normals={k:n/np.linalg.norm(n) for k,n in normals.items()}
        trials=[]
        for r in rr:
            if int(r['site'])!=site or int(r['iteration']) not in (0,1,14,15):continue
            pair=[]
            for nm in ('old_force_N','new_force_N'):
                f=np.fromstring(r[nm],sep=' ');f/=np.linalg.norm(f);angle,k=min((float(np.rad2deg(np.arccos(np.clip(n@f,-1,1)))),k) for k,n in normals.items())
                pair.append(dict(parent=k,angle_degrees=angle))
            trials.append(dict(iteration=int(r['iteration']),closest_registered_facet_directions=pair))
        records[site]=dict(cortical_body=body,nearest_registered_vertex=i,reference_distance_m=distance[i],registered_parent_facets=[list(k) for k in normals],witness=trials)
    out.write_text(json.dumps(dict(feature_report=sha(features),frame=sha(poses),side='left',sites=records),indent=2)+'\n')
if __name__=='__main__':inspect(*map(Path,sys.argv[1:]))
