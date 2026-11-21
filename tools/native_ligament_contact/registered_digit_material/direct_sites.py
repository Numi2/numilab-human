from pathlib import Path
import json,sys
import numpy as np
from material_context import get_layers,read_inputs,unit,applied
from hand_geometry import Surface
w,m,c,e=map(Path,sys.argv[1:]);r=w/'qualified/motion-008/step-8.receipt.json';pack=w/'qualified/native-coupled-006/native/resting-human.mrvpack'
layers={k:Surface(*v) for k,v in get_layers(pack,r).items()}
routes,users,bones,owners,poses=read_inputs(w/'current-package.json',m/'myosim-fullbody-reference.manifest.json',c/'emitted-native/stderr.log')
legacy=json.loads((e/'extensor-anchored-fairing-037/site-clearance.json').read_text())
prior={(i['name'],s['site_order']):np.array(s['world_m']) for g in legacy['surfaces'] for i in g['routes'] for s in i['sites']}
out=[]
for name,route in routes.items():
    for site in route[3:]:
        p=site['world'];body=site['body'];registered=owners[body]
        if len(registered)!=1:raise ValueError((name,body,registered))
        sid,bank=registered[0];b=Surface(*bones[sid]);face,q,bary,d=b.nearest(p);normal=unit(np.cross(*(np.diff(b.tri[face],axis=0))))
        if np.mean(np.einsum('ij,ij->i',b.tri[:,0]-b.v.mean(0),np.cross(b.tri[:,1]-b.tri[:,0],b.tri[:,2]-b.tri[:,0])))<0:normal=-normal
        layers_now={k:dict(inside=s.inside(p),distance_mm=1000*s.nearest(p)[3]) for k,s in layers.items()}
        # Look for the exterior skin and height of the deep surface on the same registered bone normal.
        ends={k:[round(1000*t,5) for t,_ in s.ray(q+normal*1e-8,normal,.018)] for k,s in layers.items()}
        out.append(dict(name=name,order=site['order'],site=site['site'],body=body,bank=sid,owner=bank,users=users[site['site']],world=p.tolist(),local=site['local'].tolist(),previous_error_mm=float(1000*np.linalg.norm(p-prior[name,site['order']])),bone_inside=b.inside(p),bone_surface_distance_mm=d*1000,near_bone=q.tolist(),triangle=face,bary=bary.tolist(),normal=normal.tolist(),layers=layers_now,normal_hits_mm=ends))
Path('direct-sites.json').write_text(json.dumps(out,indent=2)+'\n')
for s in out:
    if not s['layers']['outer']['inside'] or s['order']==8: print(s['name'],s['order'],round(s['previous_error_mm'],3),'bone',s['bone_inside'],round(s['bone_surface_distance_mm'],2),'in',[(k,v['inside']) for k,v in s['layers'].items()],'ray',s['normal_hits_mm'],flush=True)
