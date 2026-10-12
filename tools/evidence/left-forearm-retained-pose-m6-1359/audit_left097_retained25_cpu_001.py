#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import ast,hashlib,json,math,pickle,struct,sys,time,os,gc
import numpy as np
from fractions import Fraction
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
START=time.monotonic()
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
FD=Path('/Users/n/numi-human-fds132-m6-1350')
OUT=A/'left097-retained25-cpu-audit-001'
require=lambda v,m: None if v else (_ for _ in ()).throw(RuntimeError(m))
require(not OUT.exists(),'preserve previous attempt')
OUT.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n')
def extract(p,names,ns):
    tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes}==names,'AST helper missing')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
AUD=FD/'attempt-099-candidate097-full-source-and-late-audit-001/audit_candidate097_full_source_and_late.py'
FWD=FD/'attempt-088-registered-route-field-refined-2mm-001.py'
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
CI=Path('/Users/n/numi-human-anatomy-completion-1276/src/numilab_human/cardiac_cavity_intersections.py')
sys.path.insert(0,str(CI.parent.parent))
from numilab_human import cardiac_cavity_intersections as ci
ns={'np':np,'defaultdict':defaultdict,'ci':ci,'gc':gc}
extract(AUD,{'topology','union_mesh','lattice','audit_mesh'},ns)
fw={'np':np,'math':math,'struct':struct,'hashlib':hashlib,'Path':Path,'Fraction':Fraction}
extract(FWD,{'sha','req','F32','parse_nhrigid','rotate_cxx_batch','rotate_metal_batch','f32sum_points'},fw)
h={'np':np,'struct':struct,'Path':Path}
extract(H,{'require','load_tissue','row_data'},h)
CAND=FD/'attempt-097-minimal-late-witness-diagonal-screen-001/candidate-source-preserving-flip.pkl'
FIELD=FD/'attempt-097-minimal-late-witness-diagonal-screen-001/route-anchor-reference-field-flipped.npz'
TISS=A/'forearm-shared-field-native-012-diagonal097/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
RIG=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
ARCH=A.parent/'skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz'
pins={str(p):sha(p) for p in (Path(__file__),AUD,FWD,H,CI,CAND,FIELD,TISS,RIG,ARCH)}
require(sha(CAND)=='7b0ae1efee39b3350e64692c0e5868ab4dcd2239bcc2160966ffca839365f244','candidate drift')
C=pickle.loads(CAND.read_bytes());u=ns['union_mesh'](C);require(not u['errors'],'invalid union')
tissue=h['load_tissue'](TISS);canon={}
for sid in (132,134):
    r=h['row_data'](tissue,sid)
    for b in tissue['bindings'][r['fb']:r['fb']+r['bc']]:
        core=int(b['core']);value=b['value']
        require(core not in canon or np.array_equal(canon[core],value),'canonical mismatch')
        canon[core]=value.copy()
fw['CANON']=canon
default=fw['parse_nhrigid'](RIG,ARCH)['poses']
with np.load(FIELD,allow_pickle=False) as z:
    cores=z['core_body_indices'].copy()
    points={sid:z[f'stable{sid}_rest_positions_m'].copy() for sid in (132,134)}
    weights={sid:z[f'stable{sid}_core_weights'].copy() for sid in (132,134)}
P=A.parent/'skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011'
items=[]
for name,base,steps in [
 ('parent',A/'temporalis164-native-m6-002/baseline/native-run/accepted-geometry',[0,4767,5023,5599,6207,6815,7423,8000]),
 ('mtp',A/'mtp-passive-native-sensitivity-one-001/baseline/native-run/accepted-geometry',[8000]),
 *[(arm,P/arm/'native-run/accepted-geometry',[0,47519,151999,152607,153215,153823,154431,155000]) for arm in ('baseline','intervention')]]:
    for step in steps:
        p=base/f'step-{step}.receipt.json';pins[str(p)]=sha(p);items.append((name,step,p))
save(OUT/'inputs.json',{'pins':pins,'scope':'CPU ordinary FMA deformation at 25 retained poses, not actual GPU captures or whole-body anatomical admission.'})
results=[]
for name,step,p in items:
    j=json.loads(p.read_text())
    poses={int(b['body_index']):(np.asarray(b['position_m'],dtype='<f4'),np.asarray(b['quaternion_xyzw'],dtype='<f4')) for b in j['accepted_registered_body_poses']}
    coords={sid:fw['f32sum_points'](points[sid],weights[sid],cores,default,poses,'fma') for sid in (132,134)}
    lookup={}
    for sid in (132,134):
        for gid,xyz in zip(C['surfaces'][sid]['global_vertex_ids'],coords[sid]):
            gid=int(gid)
            require(gid not in lookup or lookup[gid].tobytes()==xyz.tobytes(),'shared F32 coordinate mismatch')
            lookup[gid]=xyz
    meshes=[(str(sid),np.asarray(C['surfaces'][sid]['faces'],dtype=np.int64),coords[sid]) for sid in (132,134)]
    meshes.append(('union',u['faces'],np.asarray([lookup[g] for g in u['gids']],dtype='<f4')))
    row={'arm':name,'step':step,'pose_sha256':pins[str(p)],'meshes':{}}
    for label,faces,xyz in meshes:
        lattice=ns['lattice'](xyz)
        deg=[i for i,(a,b,c) in enumerate(faces) if not any(ci._cross(ci._sub(lattice[int(b)],lattice[int(a)]),ci._sub(lattice[int(c)],lattice[int(a)])))]
        if deg: result={'passed':False,'degenerate_count':len(deg),'degenerate_faces':deg}
        else: result=ns['audit_mesh'](label,faces,xyz)
        top=ns['topology'](faces,len(xyz),xyz)
        result['topology']=top
        result['passed']=result['passed'] and top['closed'] and top['consistently_oriented'] and all(v['sign']!='zero' for v in top['component_signed_volumes'])
        row['meshes'][label]=result
    row['passed']=all(v['passed'] for v in row['meshes'].values())
    save(OUT/f'{name}-{step}.json',row);results.append(row)
    print(json.dumps({'arm':name,'step':step,'passed':row['passed'],'pairs':{k:v.get('exact_float32_self_pair_count') for k,v in row['meshes'].items()},'elapsed_s':time.monotonic()-START}),flush=True)
require(all(sha(p)==d for p,d in pins.items()),'input changed')
save(OUT/'report.json',{'status':'passed' if all(r['passed'] for r in results) else 'failed','results':results,'input_sha256':pins,'scope':'25 retained poses, CPU ordinary FMA field plus exact Float32-lattice self/degeneracy/topology checks. Not native GPU or external-interface qualification. Native012 measured predictor discrepancy prevents a bitwise GPU claim.','gpu_used':False,'anatomical_admission_claimed':False,'elapsed_seconds':time.monotonic()-START})

