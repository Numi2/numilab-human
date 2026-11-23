"""Test bilateral shoulder fiber reachability on sections through the actual registered cortical surfaces."""
from pathlib import Path
import ctypes,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from audit_fiber_load_bounds import endpoints,read_pose
from compile_connected_joint import digest
from rebind_ligament_material import banks


def sections(tris,o,axis,bend):
    normal=np.cross(axis,bend)
    walls=[];owners=[]
    for sid,tri in tris.items():
        v=tri-o
        z=v@normal
        near=(z.min(1)<0)&(z.max(1)>0)
        if not near.any():continue
        k=v[near];m=z[near]
        # Interpolate the two crossed cortical edges, retaining all section walls for solid membership.
        j,edge=np.where((m>=0)!=(np.roll(m,-1,axis=1)>=0))
        if len(j)!=2*len(k):raise ValueError('Indeterminate cortical section')
        nxt=(edge+1)%3
        a=k[j,edge];b=k[j,nxt]
        s=m[j,edge]/(m[j,edge]-m[j,nxt])
        v=a+s[:,None]*(b-a)
        projected=np.column_stack((v@axis,v@bend)).reshape(-1,2,2)
        walls.append(projected);owners.extend([sid]*len(k))
    return np.ascontiguousarray(np.concatenate(walls)),np.array(owners,'i4')


def route(solver,tris,a,b,limit,theta):
    length=np.linalg.norm(b-a);axis=(b-a)/length
    direction=np.eye(3)[np.argmin(abs(axis))]
    x=np.cross(axis,direction);x/=np.linalg.norm(x)
    bend=x*np.cos(theta)+np.cross(axis,x)*np.sin(theta)
    walls,owners=sections(tris,a,axis,bend)
    p=walls.reshape(-1,2)
    end=np.array([length,0.])
    p=p[np.linalg.norm(p,axis=1)+np.linalg.norm(p-end,axis=1)<=limit+1e-10]
    _,indices=np.unique(np.round(p,9),axis=0,return_index=True)
    p=p[np.sort(indices)]
    p=np.ascontiguousarray(np.vstack(([0.,0.],end,p)))
    result=np.empty(len(p),'i4')
    count=solver(p,len(p),walls,owners,len(walls),limit,result)
    if count<0:raise ValueError('Invalid cortical section')
    if not count:return None
    p=p[result[:count]]
    return a+p[:,:1]*axis+p[:,1:]*bend


def run(parent,source,author,output):
    bone=next((parent/'supported-bones').glob('*.nhbones'))
    posefile=parent/'native-ligament-material/stderr.log'
    pose=read_pose(posefile)
    material=parent/'registered-ligaments/registration.json'
    reg=json.loads(material.read_text())
    if pose['bonePayloadSha256']!=digest(bone) or reg['candidate']!=digest(bone):raise ValueError('Different native body generation')
    sur=banks(bone);pp=pose['bodyPoses'];all_tris={}
    for (body,sid),(f,v,_) in sur.items():
        q=pp[body];all_tris[sid]=(Rotation.from_quat(q[3:]).apply(v)+q[:3])[f]*1000
    models=json.loads((source/'source-routes.json').read_text())['routes']
    projections=json.loads((source/'registered-wraps.json').read_text())
    from numpy.ctypeslib import ndpointer
    lib=ctypes.CDLL(str(author/'section.dylib'));solver=lib.section_path
    ft=ndpointer(np.float64,flags='C_CONTIGUOUS');it=ndpointer(np.int32,flags='C_CONTIGUOUS')
    solver.argtypes=[ft,ctypes.c_int,ft,it,ctypes.c_int,ctypes.c_double,it];solver.restype=ctypes.c_int
    todo=[]
    for model,projection in zip(models,projections,strict=True):
        if model['name']!=projection['name']:raise ValueError('Mixed muscle model sources')
        side=int(model['side']=='r')
        name=('superior_glenohumeral_capsuloligament' if model['name'].startswith('s_gleno') else 'coracohumeral_capsuloligament')+'/'+str(side)
        sites=[]
        for a,b in zip(projection['projected_sites'],model['sites'],strict=True):
            if a['stable_id']!=b['stable_id']:raise ValueError('Different cortical owner')
            sites.append(dict(body=b['body'],bone=b['stable_id'],triangle=a['face'],barycentric=a['barycentric']))
        todo.append((name,sites,model['source_springlength_m'][0],'MyoSim registered landmarks and source spring-length parameter'))
    for side in (0,1):
        name='coracoclavicular_complex/'+str(side);v=reg['fibers'][name]
        todo.append((name,v['endpoints'],v['reference_length_m'],'transported native material attachments and original reference length'))
    output.mkdir()
    for name,sites,ref,provenance in todo:
        a,b=endpoints(dict(endpoints=sites),sur,pp)*1000
        # Every admissible path shorter than the limit lies within this ellipsoid and its bounding box.
        limit=max(np.linalg.norm(b-a)+7.,ref*1000+12.)
        center=(a+b)*.5;radius=limit*.5
        tris={s:v for s,v in all_tris.items() if np.all(v.min((0,1))<=center+radius) and np.all(v.max((0,1))>=center-radius)}
        best=None;samples=[]
        for angle in np.arange(20)*np.pi/20+.000183:
            p=route(solver,tris,a,b,limit,angle)
            if p is None:samples.append(None);continue
            L=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            samples.append(L)
            if best is None or L<best[0]:best=L,p,float(angle);limit=min(limit,L+.00005)
        rec=dict(name=name,source=provenance,sites=sites,bones=digest(bone),pose=digest(posefile),source_site_table=digest(source/'registered-wraps.json'),source_law=digest(source/'source-routes.json'),native_fibers=digest(material),reference_m=ref,chord_m=float(np.linalg.norm(b-a)*.001),best_length_m=best[0]*.001 if best else None,world_route_m=(best[1]*.001).tolist() if best else [],angle_rad=best[2] if best else None,broadphase_bones=sorted(tris),sampled_section_upper_bounds_mm=samples)
        (output/(name.replace('/','-')+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
        print(name,round(rec['chord_m']*1000,2),round(rec['best_length_m']*1000,2) if best else None,1000*ref,len(best[1]) if best else 0,flush=True)
if __name__=='__main__':run(*[Path(f).resolve() for f in sys.argv[1:]])
