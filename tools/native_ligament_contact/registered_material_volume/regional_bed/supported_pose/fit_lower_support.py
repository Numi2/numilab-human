"""Find a gravity-loaded starting posture using native material sections and actual skin Jacobians."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
def read(p):return json.loads(p.read_bytes())
def records(p):
    with p.open() as s:return list(csv.DictReader(s))
def run(source,measured,out):
    a=read(source/'launch.json')
    env=a['environment'];base=source/'native-traces'
    if read(measured/'execution.json')['exit']:raise ValueError('Incomplete observation')
    m=read(measured/'launch.json')
    if m['argv'][1:4]!=a['argv'][1:4] or env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':
        raise ValueError('Different skeletal source')
    q=records(base/'resting-com-q-integration.csv')[0]
    other=records(measured/'native-traces/resting-com-q-integration.csv')[0]
    if q['q_before_fingerprint_fnv64']!=other['q_before_fingerprint_fnv64']:raise ValueError('Different initial body configuration')
    lines=(measured/'stdout.log').read_text().splitlines()
    target=next(json.loads(l.split('=',1)[1])for l in lines if l.startswith('initial_gravity_total='))
    force=next(json.loads(l.split('=',1)[1]) for l in (source/'stdout.log').read_text().splitlines()if l.startswith('persistent_initial_force_reference='))
    rows=[json.loads(l.split('=',1)[1])for l in lines if l.startswith('initial_bed_posture_row=')]
    bed=records(base/'resting-local-bed-sections.csv')
    normal=records(base/'resting-com-support-impulses.csv')[0]
    n=np.array([float(normal['normal_'+k]) for k in 'xyz'])
    g=np.array([0,0,9.81])*target['mass_kg']
    com=np.array(target['center_m']);v=np.array([r['xyz'] for r in rows]);gap=np.array([r['gap']for r in rows]);J=np.array([r['Jnormal']for r in rows])
    c=np.array([float(b['skin_compliance_m_per_n'])+float(b['fat_compliance_m_per_n'])for b in bed])
    stock=np.array([float(b['skin_thickness_m'])+float(b['fat_thickness_m'])for b in bed])
    if len(rows)!=32 or any(r['row']!=i or r['source']!=int(bed[i]['source_geometry_index']) for i,r in enumerate(rows)):raise ValueError('Mismatched skin material rows')
    # These independent sagittal coordinates move the femur, with the original coupled knee untouched.
    dofs=[100,114];initial_q=np.fromstring(q['q_before_f32_semicolon'],sep=';')
    qr=records(base/'resting-com-q-index-map.csv')
    qids=[int(next(r['local_q_index'] for r in qr if r['local_v_index']==str(k) and r['local_q_index']))for k in dofs]
    if qids!=[101,115] or any(abs(initial_q[qids])>1e-6):raise ValueError('Incompatible hip indexing or opening angle')
    eq=Path(a['argv'][a['argv'].index('--joint-equality-payload')+1]).read_bytes()
    import struct
    count=struct.unpack_from('<I',eq,20)[0]
    if len(eq)!=80+count*96 or any(struct.unpack_from('<I',eq,80+i*96)[0] in qids for i in range(count)):raise ValueError('A dependent joint was requested')
    demand=np.array([next(r['gravity_target_n']for r in force['rows']if r['dof']==k)for k in dofs])
    def evals(s):
        h,rx,ry,u,w=s
        moved=Rotation.from_rotvec([rx,ry,0]).apply(v-com)+com+h*n
        delta=J[:,dofs]@np.array([u,w])
        # The normal projection is observed for every row, including mixed donor rows.
        dp=(moved-v)@n+delta
        loads=np.maximum(0,-(gap+dp))/c
        moments=np.cross(moved-com,n) * loads[:,None]
        residual=np.r_[loads.sum()-n@g,moments.sum(0)[:2],J[:,dofs].T@loads-demand]
        return residual,loads
    def obj(s):return evals(s)[0]
    sols=[]
    for angle in [-.026,-.035,-.05]:
        q0=np.array([.001,.003,0,angle,angle])
        sol=least_squares(obj,q0,x_scale=[.002,.01,.01,.03,.03],bounds=([-.016,-.055,-.055,-.105,-.105],[.016,.055,.055,.035,.035]),max_nfev=250,gtol=1e-11,ftol=1e-13,xtol=1e-13)
        res,loads=evals(sol.x)
        sols.append((np.linalg.norm(res),sol.x.tolist(),res.tolist(),loads))
    norm,s,res,loads=min(sols,key=lambda i:i[0]);h,x,y,u,w=s
    if norm>.2 or np.max(loads*c/stock)>.98:raise ValueError(('Unbalanced construction',norm,res,s))
    if min(loads[[10,20,21]].sum(),loads[[11,30,31]].sum())<1:raise ValueError('No bilateral lower limb support')
    i=a['argv'].index('--root-pose')+1;root=read(source/'native-initial-bodies.json')['all_bodies'][0]['current']
    R=Rotation.from_rotvec([x,y,0]);xyz=com+R.apply(np.array(root['position_world_m'])-com)+h*n;quat=R*Rotation.from_quat(root['orientation_world_xyzw'])
    a['argv'][i:i+7]=list(map(str,np.r_[xyz,quat.as_quat()]));a['argv']+=['--pose-q',str(qids[0]),str(u),'--pose-q',str(qids[1]),str(w)]
    out.mkdir();f=out/'construction.json'
    f.write_text(json.dumps(dict(launch=digest(source/'launch.json'),jacobian_source=digest(measured/'stdout.log'),q_hash=q['q_before_fingerprint_fnv64'],center=com.tolist(),initial_gap_m=gap.tolist(),joint_dofs=dofs,joint_q=qids,hip_delta_rad=[u,w],
                    translation_along_normal_m=h,world_rotation_vector=[x,y,0],gravity_moment_Nm=demand.tolist(),prediction_force_N=loads.tolist(),residual=res,max_stock_fraction=float(np.max(loads*c/stock)),original_residual=evals(np.zeros(5))[0].tolist()),indent=2)+'\n')
    a['pins'][str(f.resolve())]=digest(f);a['lower_limb_initial_posture']=digest(f)
    (out/'launch.json').write_text(json.dumps(a,indent=2)+'\n')
    print('fit',norm,'angles deg',np.rad2deg([x,y,u,w]),'lift mm',h*1000,'loads',[(k,round(n,1))for k,n in enumerate(loads) if n>0],flush=True)
if __name__=='__main__':run(*(Path(p).resolve()for p in sys.argv[1:]))
