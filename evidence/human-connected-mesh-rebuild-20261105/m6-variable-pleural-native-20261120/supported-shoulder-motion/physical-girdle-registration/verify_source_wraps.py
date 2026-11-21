"""Use the source tendon evaluator to test material projections against the registered shoulder bones."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from source_pair_audit import inputs
from shoulder_inputs import old,digest
def closest(tri,x):
    a,b,c=tri
    ab,ac=b-a,c-a
    uv=np.linalg.lstsq(np.stack((ab,ac),axis=1),x-a,rcond=None)[0]
    w=np.r_[1-uv.sum(),uv];options=[]
    if min(w)>=0:options.append(w)
    for j in range(3):
        k=(j+1)%3;e=tri[k]-tri[j];u=np.clip(np.dot(x-tri[j],e)/np.dot(e,e),0,1)
        v=np.zeros(3);v[j]=1-u;v[k]=u;options.append(v)
    best=min(options,key=lambda v:np.linalg.norm(v@tri-x))
    # Keep each root inside its source triangle, including an edge projection.
    return best*(1-3e-5)+1.e-5
def fmt(v):return ' '.join('%.16g'%x for x in v)
def source_model(h,s,w):
    c=w['center_local_m'];ss=w['side_site_local_m'];r=w['radius_m']
    xml=f"""<mujoco><worldbody>
      <body name="hum"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <geom name="wrap" type="sphere" size="{r}" pos="{fmt(c)}"/>
      <site name="start" pos="{fmt(h)}"/><site name="side" pos="{fmt(ss)}"/></body>
      <body name="scap"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <site name="end" pos="{fmt(s)}"/></body></worldbody><tendon><spatial name="source">
      <site site="start"/><geom geom="wrap" sidesite="side"/><site site="end"/></spatial></tendon></mujoco>"""
    m=mujoco.MjModel.from_xml_string(xml);return m,mujoco.MjData(m)
def path(m,d,ph,ps,c):
    for i,p in [(0,ph),(7,ps)]:
        d.qpos[i:i+3]=p[:3]; d.qpos[i+3:i+7]=p[[6,3,4,5]]
    mujoco.mj_forward(m,d);count=int(d.ten_wrapnum[0]);assert count in (2,4)
    points=d.wrap_xpos.reshape(-1,3)[:count].copy();length=float(d.ten_length[0])
    if count==2:return points,length,0.
    center=R.from_quat(ph[3:]).apply(c)+ph[:3]
    a,b=points[1:3]-center;r=np.linalg.norm(a);u=a/r
    axis=np.cross(a,b);axis/=np.linalg.norm(axis)
    angle=(length-np.linalg.norm(points[0]-points[1])-np.linalg.norm(points[3]-points[2]))/r
    if np.linalg.norm(R.from_rotvec(axis*angle).apply(a)-b)>1e-7:axis=-axis
    assert np.linalg.norm(R.from_rotvec(axis*angle).apply(a)-b)<1e-7
    n=max(1,int(np.ceil(angle/.024)))
    arc=center+R.from_rotvec(np.outer(np.linspace(0,angle,n+1),axis)).apply(a)
    return np.vstack([points[0],arc,points[-1]]),length,float(r*(1-np.cos(angle/n/2)))
def run(launch,ref,motion,bones,source,out):
    _,poses,rest,names=inputs(launch,ref,motion);poses={0:rest,**poses}
    old.audit.ag.BONES=bones;_,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
    ms=old.load_local_meshes(raw,cat,vo,io);d=json.loads(source.read_text());assert d['bone_sha256']==digest(bones)
    results=[]
    for route in d['routes']:
        sites=route['sites'];members=[s['stable_id'] for s in sites];sides=members+[9 if route['side']=='l' else 8]
        real=[np.asarray(s['point_local_m']) for s in sites];weights=[];projected=[]
        for s in sites:
            v,f,*_=ms[s['stable_id']];face=s['geometry']['registered_candidate']['face'];tri=v[f[face]]
            w=closest(tri,s['point_local_m']);weights.append(w);projected.append(w@tri)
        geometry={}
        for sid in sides:
            v,f,*_=ms[sid];t=v[f];geometry[sid]=(t,t.min(1),t.max(1))
        shape=route['source_wrap'];c=shape['center_local_m'];src=source_model(*real,shape);proj=source_model(*projected,shape)
        measures=[];first=None; maximum_sag=0.;hit_counts={k:0 for k in sides}
        for k,p in poses.items():
            ph,ps=[p[s['body']] for s in sites]
            _,l0,_=path(*src,ph,ps,c)
            if k==0:assert abs(l0-route['default_source_path_m'])<1e-7
            pts,l,sag=path(*proj,ph,ps,c);maximum_sag=max(sag,maximum_sag)
            hits=[]
            for sid in sides:
                pose=p[cat[sid]['body']];loc=R.from_quat(pose[3:]).inv().apply(pts-pose[:3])
                tri,lo,hi=geometry[sid]
                for j,(a,b) in enumerate(zip(loc[:-1],loc[1:])):
                    mask=np.all((lo<=np.maximum(a,b)+1e-9)&(hi>=np.minimum(a,b)-1e-9),axis=1)
                    if not np.any(mask):continue
                    selected=np.flatnonzero(mask);t=tri[selected].reshape(-1,3)
                    h=old.audit.lp.vector_hits(a,b,(t,np.arange(len(t)).reshape(-1,3)))
                    if h:hits.append([sid,j,int(selected[h[0][1]])]);break
            for sid,_,_ in hits:hit_counts[sid]+=1
            if first is None and hits:first=dict(step=k,segment_bone_faces=hits)
            measures.append([k,l,l0,len(pts)>2,bool(hits)])
        m=np.asarray(measures)
        results.append(dict(name=route['name'],poses=len(m),projected_sites=[dict(stable_id=s['stable_id'],face=s['geometry']['registered_candidate']['face'],barycentric=w.tolist(),displacement_m=float(np.linalg.norm(q-v))) for s,w,q,v in zip(sites,weights,projected,real)],
            wrap_active_poses=int(m[:,3].sum()),route_crossing_poses=int(m[:,4].sum()),witness=first,poses_with_hits_by_bone=hit_counts,
            max_arc_sampling_sag_m=maximum_sag,maximum_source_path_length_difference_m=float(np.max(np.abs(m[:,2]-m[:,1]))),
            minimum_projected_length_m=float(m[:,1].min()),maximum_projected_length_m=float(m[:,1].max())))
        print(route['name'],results[-1]['wrap_active_poses'],results[-1]['route_crossing_poses'],first,flush=True)
        out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
