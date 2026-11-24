"""Separate exact anatomical bone attachment sites from the auxiliary frozen muscle surface nodes."""
import json,sys
from pathlib import Path
import igl
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_shoulder import read,LABELS,MOVING
from assemble_material import frames
from measure_assembly_contacts import canonical
from fit_registered_assembly import layer_surface
from compile_connected_joint import digest

def run(j,reg,fit,emitted,out):
    case=j/'posterior-interface-082/native-pair';a=read(case/'launch.json')['argv'];pose=frames(case)[-1]
    mr=Path(a[a.index('--tendon-payload')+1]).with_suffix('.manifest.json');ends=read(mr)['endpoints'];org=np.array(read(reg/'registration.json')['origin'])
    sites=[];allxyz=[]
    for sid in MOVING:
        z=np.load(emitted/f'bank-{sid}.npz');p=(z['native']-org)*1000;allxyz.append(p)
        for v in np.load(fit/f'{sid}.npz')['fixed_nodes']:
            if v<len(p):sites.append((sid,'fixed_material_vertex',int(v),None,None,p[v]))
        for e in ends:
            if e.get('muscle')!=LABELS[sid]:continue
            b=pose[e['body_index']];surface=e.get('surface') or {};raw=[e['resolved_local_point_m']]+surface.get('node_local_points_m',[])
            world=(Rotation.from_quat(b['quaternion_xyzw']).apply(raw)+b['position_m']-org)*1000
            for i,x in enumerate(world):sites.append((sid,'tendon_endpoint' if i==0 else 'distributed_bone_site',i,e['endpoint'],surface.get('bone_stable_id'),x))
    pts=np.asarray([v[-1]for v in sites]);q=np.vstack(allxyz)
    solids=layer_surface(j,org,np.minimum(q.min(0),pts.min(0)),np.maximum(q.max(0),pts.max(0)))
    refs=np.load(reg/'neighbors.npz')
    for key in refs.files:
        if not key.endswith('-v'):continue
        kind,bank=map(int,key[:-2].split('-'))
        if kind!=51004:continue
        x=refs[key];l,h=x.min(0),x.max(0)
        if not np.any(np.all((pts>=l)&(pts<=h),axis=1)):continue
        s,_=canonical(x,refs[key[:-1]+'f'])
        if s is None:raise ValueError('Open bone in attachment footprint')
        solids[kind,bank]=s,l,h
    rows=[]
    for (kind,bank),(solid,l,h) in solids.items():
        ids=np.flatnonzero(np.all((pts>=l)&(pts<=h),axis=1))
        if not len(ids):continue
        m=solid.to_mesh64();v=np.ascontiguousarray(m.vert_properties[:,:3]);f=np.ascontiguousarray(m.tri_verts,dtype='i8')
        d,_,nearest=igl.point_mesh_squared_distance(pts[ids],v,f)
        w=np.abs(igl.winding_number(v,f,pts[ids]))
        for ind,d2,wind,near in zip(ids,d,w,nearest):
            dist=np.sqrt(d2)
            if dist<.05:continue
            if min(abs(wind),abs(1-wind))>1e-6:raise ValueError('Ambiguous winding away from interface')
            if wind<.5:continue
            sid,cls,idx,endpoint,own,x=sites[ind]
            rows.append(dict(muscle=sid,site_kind=cls,site=idx,endpoint=endpoint,bound_bone=own,obstacle_kind=kind,bank=bank,distance_inside_mm=float(dist),winding=float(wind),point_mm=x.tolist(),cortical_or_tissue_exit_mm=near.tolist()))
    result=dict(native=digest(case/'launch.json'),manifest=digest(mr),fit=digest(fit/'result.json'),geometry=digest(emitted/'results.json'),
        total_checked_sites=len(sites),strict_margin_mm=.05,cases=rows)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for k in ['fixed_material_vertex','tendon_endpoint','distributed_bone_site']:
        v=[r for r in rows if r['site_kind']==k and r['obstacle_kind']==0]
        print(k,len(v),sorted([(r['muscle'],r['bank'],round(r['distance_inside_mm'],3))for r in v],key=lambda a:-a[2])[:6])
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:]))
