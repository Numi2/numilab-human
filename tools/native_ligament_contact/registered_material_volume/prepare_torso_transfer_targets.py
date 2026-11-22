"""Collect source-corresponding torso material witnesses in the matched release pose."""
import json,sys,runpy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from emit_skin_source_registration import sha
launch,obs,freeze,envout,owner,human,out=map(Path,sys.argv[1:])
from importlib.util import spec_from_file_location,module_from_spec
sys.path.insert(0,str(owner/'tools'));sys.path.insert(0,str(human/'src'))
from rebind_human_skin_reference import decode
from numilab_human.myosim_bone_proximity import _point_triangle_distances_squared,_closest_point_on_triangle
j=json.loads(launch.read_text());env=j['environment'];ref=json.loads(freeze.read_text())
asset=Path(j['argv'][j['argv'].index('--skin-payload')+1])
assert sha(asset)==ref['current']['sha256'] and sha(obs)==ref['current_capture_sha256']
src=np.loadtxt(obs,skiprows=1,delimiter=',',usecols=(2,3,4)).astype('f4').astype('f8')
sface=decode(asset.read_bytes())['inds'].reshape(-1,3)
tri=src[sface];centers=tri.mean(1);radii=np.linalg.norm(tri-centers[:,None,:],axis=2).max(1)
classes=[np.flatnonzero(np.frexp(radii)[1]==ex) for ex in np.unique(np.frexp(radii)[1])]
trees=[cKDTree(centers[k]) for k in classes];max_r=[radii[k].max() for k in classes]
v_tree=cKDTree(src)
pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
outer=np.loadtxt(env[pfx+'NODES_CSV'],skiprows=1,delimiter=',',usecols=(2,3,4)).astype('f4').astype('f8')
faces=np.loadtxt(env[pfx+'FACES_CSV'],skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4')
attribution=json.loads((envout/'attribution.json').read_text())
ids={q['stable_id'] for q in attribution['source_comparison'] if q['source']['intersection_pairs']==0}
if ids!={67,68,73,74}:raise ValueError('The qualified clean source targets changed')
rows=np.unique([p[0] for row in map(json.loads,(envout/'target-witnesses.jsonl').open()) if row['semantic']==51005 and row['stable_id'] in ids for p in row['triangle_pairs']])
nodes=np.unique(faces[rows]);polygon=outer[faces[rows]]
orig=np.vstack((outer[nodes],polygon.mean(axis=1),*((polygon[:,i]+polygon[:,(i+1)%3])*.5 for i in range(3))))
bound,_=v_tree.query(orig);closest=[];source_ids=[];weights=[]
for p,d0 in zip(orig,bound):
    groups=[k[t.query_ball_point(p,d0+radius+1e-12)] for k,t,radius in zip(classes,trees,max_r)]
    candidates=np.unique(np.concatenate(groups))
    dist=_point_triangle_distances_squared(p,tri[candidates],np)
    nearest=int(candidates[np.argmin(dist)]);q,w=_closest_point_on_triangle(p,tri[nearest],np)
    if abs(np.dot(q-p,q-p)-dist.min())>2e-13 or np.linalg.norm(q-p)>d0+1e-10:raise ValueError('Source projection certificate failed')
    closest.append(q);source_ids.append(nearest);weights.append(w)
target=np.asarray(closest);disp=target-orig
out.mkdir(exist_ok=True)
np.savez_compressed(out/'targets.npz',skin_face_rows=rows,surface_node=nodes,source_face=source_ids,source_weights=weights,start_m=orig,end_m=target)
d=dict(scope='measured source-surface targets for an offline volume fit',source_skin_sha256=sha(asset),material_surface_sha256=env[pfx+'NODES_SHA256'],source_capture_sha256=sha(obs),targets_sha256=sha(out/'targets.npz'),surface_rows=len(nodes),material_faces=len(rows),matched_source_muscles=sorted(ids),bounds_m=[orig.min(0).tolist(),orig.max(0).tolist()],vertex_distance_quantiles_m=np.quantile(np.linalg.norm(disp[:len(nodes)],axis=1),[0,.5,.9,.99,1]).tolist(),interior_distance_quantiles_m=np.quantile(np.linalg.norm(disp[len(nodes):],axis=1),[0,.5,.9,.99,1]).tolist(),median_axis_displacement_m=np.median(disp,axis=0).tolist())
(out/'target-summary.json').write_text(json.dumps(d,indent=2)+'\n')
