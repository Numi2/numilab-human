"""Use independent spine and bilateral hip coordinates, including all NHEQ followers."""
from pathlib import Path
import json,struct,sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fit_lower_support import records,read
from compile_connected_joint import digest
def main(base,measured,prior,out):
    src=read(base/'launch.json');a=src['argv']; report=read(prior/'construction.json')
    logs=(measured/'stdout.log').read_text().splitlines()
    rows=[json.loads(l.split('=',1)[1])for l in logs if l.startswith('initial_bed_posture_row=')]
    t=next(json.loads(l.split('=',1)[1]) for l in logs if l.startswith('initial_gravity_total='))
    force=next(json.loads(l.split('=',1)[1])for l in (base/'stdout.log').read_text().splitlines()if l.startswith('persistent_initial_force_reference='))
    if report['jacobian_source']!=digest(measured/'stdout.log') or report['launch']!=digest(base/'launch.json'):raise ValueError('Mismatched loaded state')
    G=np.array([v['gravity_target_n']for v in sorted(force['rows'],key=lambda r:r['dof'])])
    tr=base/'native-traces'
    state=records(tr/'resting-com-q-integration.csv')[0];q=np.fromstring(state['q_before_f32_semicolon'],sep=';')
    com=np.array(t['center_m']);x=np.array([r['xyz']for r in rows]);g=np.array([r['gap']for r in rows]);J=np.array([r['Jnormal']for r in rows])
    b=records(tr/'resting-local-bed-sections.csv')
    c=np.array([float(r['skin_compliance_m_per_n'])+float(r['fat_compliance_m_per_n'])for r in b])
    thick=np.array([float(r['skin_thickness_m'])+float(r['fat_thickness_m'])for r in b]);normal=records(tr/'resting-com-support-impulses.csv')[0];n=np.array([float(normal['normal_'+k])for k in 'xyz'])
    eq=Path(a[a.index('--joint-equality-payload')+1]).read_bytes();nr=struct.unpack_from('<I',eq,20)[0]
    if len(eq)!=80+nr*96:raise ValueError('Invalid equality source')
    mapping=records(tr/'resting-com-q-index-map.csv')
    vm={int(r['local_q_index']):int(r['local_v_index'])for r in mapping if r['local_q_index'] and r['local_v_index']}
    qs=[7,101,115];nv=len(G);E=np.zeros((nv,3));E[[vm[k]for k in qs],range(3)]=1
    for i in range(nr):
        y,v,z,iz,y0,x0,a0,a1,a2,a3,a4,_=struct.unpack_from('<4I8f',eq,80+i*96)
        if z==0xffffffff:continue
        delta=q[z]-x0
        E[v]=(a1+2*a2*delta+3*a3*delta**2+4*a4*delta**3)*E[iz]
    P=J@E; want=G@E
    def evaluate(s):
        h,rx,ry,*coord=s
        v=Rotation.from_rotvec([rx,ry,0]).apply(x-com)+com+h*n
        gap=g+(v-x)@n+P@coord; f=np.maximum(0,-gap)/c
        moments=(np.cross(v-com,n)*f[:,None]).sum(0)
        return np.r_[f.sum()-9.81*t['mass_kg']*n[2],moments[:2],P.T@f-want],f
    seeds=[(.03,.03),(.02,.05),(.04,.06),(.025,.07)]
    results=[]
    for pitch,spine in seeds:
        lo=[-.025,-.07,-.025,-.13,-.0017,-.0017];hi=[.025,.07,.025,.13,.0017,.0017]
        rr=least_squares(lambda s:evaluate(s)[0],[0,pitch,0,spine,0,0],bounds=(lo,hi),x_scale=[.005,.035,.012,.05,.001,.001],max_nfev=600,ftol=1e-13,xtol=1e-13,gtol=1e-10)
        residual,loads=evaluate(rr.x);results.append((np.linalg.norm(residual),rr.x,residual,loads))
    norm,s,r,loads=min(results,key=lambda k:k[0]);print('residual N,Nm',r.round(2),'angles deg',np.rad2deg(s[1:]).round(3),'active',[(i,round(f,2))for i,f in enumerate(loads)if f>0],flush=True)
    if max(abs(r[:3]))>6 or np.max(loads*c/thick)>.96 or min(loads[[10,20,21]].sum(),loads[[11,30,31]].sum())<20:raise ValueError('Insufficient bilateral gravity support')
    src['argv']=a.copy();a=src['argv'];i=a.index('--root-pose')+1;root=read(base/'native-initial-bodies.json')['all_bodies'][0]['current']
    rot=Rotation.from_rotvec([s[1],s[2],0]);xyz=com+rot.apply(np.array(root['position_world_m'])-com)+s[0]*n;quat=rot*Rotation.from_quat(root['orientation_world_xyzw']);a[i:i+7]=list(map(str,np.r_[xyz,quat.as_quat()]))
    for k,delta in zip(qs,s[3:]):a.extend(['--pose-q',str(k),str(q[k]+delta)])
    out.mkdir();p=out/'construction.json';p.write_text(json.dumps(dict(observation=report['jacobian_source'],source=digest(base/'launch.json'),q=qs,deltas_rad=s[3:].tolist(),translation_m=s[0],world_rotation=s[1:3].tolist(),rows=loads.tolist(),force_and_moment_residual=r.tolist(),native_gravity_moment_Nm=want.tolist(),maximum_stock_fraction=float(np.max(loads*c/thick))),indent=2)+'\n')
    src['pins'][str(p.resolve())]=digest(p);(out/'launch.json').write_text(json.dumps(src,indent=2)+'\n')
if __name__=='__main__':main(*(Path(p).resolve()for p in sys.argv[1:]))
