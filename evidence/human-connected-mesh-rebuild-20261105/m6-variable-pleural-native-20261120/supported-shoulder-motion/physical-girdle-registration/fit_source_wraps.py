"""Search source-centered material footprints without changing the donor sphere or skeletal poses."""
from pathlib import Path
import json,sys
import mujoco,numpy as np
from scipy.spatial.transform import Rotation as R
from source_pair_audit import inputs
from shoulder_inputs import old,digest,unit
from verify_source_wraps import source_model,path,closest

def run(launch,ref,motion,bones,source,out):
    _,poses,rest,names=inputs(launch,ref,motion);poses={0:rest,**poses}
    old.audit.ag.BONES=bones;_,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
    ms=old.load_local_meshes(raw,cat,vo,io);src=json.loads(source.read_text())
    results=[]
    for route in src['routes']:
        sites=route['sites'];mb=[s['stable_id'] for s in sites];active=mb+[9 if route['side']=='l' else 8]
        shape=route['source_wrap'];c=shape['center_local_m']
        m,d=source_model(*[s['point_local_m'] for s in sites],shape)
        sid=[m.site('start').id,m.site('end').id]
        rows=[];surfs={}
        for b in active:
            v,f,n,*_=ms[b];t=v[f];surfs[b]=(t,t.min(1),t.max(1))
        for site in sites:
            v,f,n,*_=ms[site['stable_id']];t=v[f];x=np.array(site['point_local_m'])
            r=np.linalg.norm(np.maximum(0.,np.maximum(t.min(1)-x,x-t.max(1))),axis=1)
            choices=[]
            for ix in np.flatnonzero(r<=.006):
                for w in [closest(t[ix],x),np.full(3,1/3)]:
                    p=w@t[ix];delta=np.linalg.norm(p-x)
                    if delta>.006:continue
                    normal=unit(np.cross(t[ix,1]-t[ix,0],t[ix,2]-t[ix,0]))
                    choices.append((int(ix),w,p,normal,unit(w@n[f[ix]]),delta))
            rows.append(choices)
        candidates=[(a[-1]**2+b[-1]**2,i,j) for i,a in enumerate(rows[0]) for j,b in enumerate(rows[1])]
        candidates.sort()
        checked=0;passed=0;winner=None
        prep={}
        for k,p in poses.items():
            pp=[p[cat[b]['body']] for b in active]
            ranges=[(R.from_quat(z[3:]),z[:3]) for z in pp]
            prep[k]=(pp,ranges)
        def evaluate(poses_to_check,intersection):
            lengths=[];wraps=0
            for step in poses_to_check:
                pp,rt=prep[step]
                points,length,sag=path(m,d,*pp[:2],c)
                for endpoint,i0,i1 in [(0,0,1),(1,-1,-2)]:
                    dr=rt[endpoint][0].inv().apply(points[i1]-points[i0])
                    if min(dr@pair[endpoint][3],dr@pair[endpoint][4])<=1.e-8:return None
                if intersection:
                    for b,(rot,tr) in zip(active,rt):
                        loc=rot.inv().apply(points-tr);t,lo,hi=surfs[b]
                        slab=np.all((lo<=loc.max(0)+1e-8)&(hi>=loc.min(0)-1e-8),axis=1)
                        if not np.any(slab):continue
                        tri=t[slab];pts=tri.reshape(-1,3);face=np.arange(len(pts)).reshape(-1,3)
                        if any(old.audit.lp.vector_hits(a,b,(pts,face)) for a,b in zip(loc[:-1],loc[1:])):return None
                lengths.append(length);wraps+=len(points)>2
            return [min(lengths),max(lengths),int(wraps)]
        for score,i,j in candidates:
            pair=[rows[0][i],rows[1][j]]
            for ix,x in zip(sid,pair):m.site_pos[ix]=x[2]
            if not evaluate([0,-1,-2,-3,32,8000],False):continue
            checked+=1
            if not evaluate([0,-1,-2,-3,32,8000],True):continue
            passed+=1
            valid=evaluate(poses,True)
            if valid:
                winner=dict(sites=[dict(face=x[0],barycentric=x[1].tolist(),source_displacement_m=x[5]) for x in pair],length_range_m=valid[:2],wraps=valid[2]);break
        results.append(dict(name=route['name'],max_radius_m=.006,material_pair_count=len(candidates),passed_outward_at_sentinels=checked,
            passed_crossings_at_sentinels=passed,result=winner))
        print(results[-1],flush=True);out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
