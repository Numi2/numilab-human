"""Bind the registered physical current pose while preserving its separately qualified rest material."""
import json,os,re,sys,subprocess
from pathlib import Path
import numpy as np
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks
from certify_cortical_paths import Cortex,verify
from compile_connected_joint import digest
from prepare_live_cortex import fl,exact
from run_cavity_motion import execute_native
from certify_transported_material import measure
from scipy.spatial.transform import Rotation

def run(h,build,baseline,fk,field,out):
    out.mkdir();model=out/'app';model.mkdir()
    src=build/'app';main='numilab_human_myosim_visual_probe.mm';unit='NumiHumanLoadedSkinPoseData.inc';reg='NumiHumanCorticalRegistration.inc'
    for f in src.iterdir():
        if f.name not in [main,unit,reg] and f.suffix in ('.mm','.hpp','.inc'):(model/f.name).symlink_to(f.resolve())
    # Copy the ligament header: its sibling generated coordinate catalog must resolve here.
    (model/'NumiHumanRegisteredLigaments.hpp').unlink();(model/'NumiHumanRegisteredLigaments.hpp').write_bytes((src/'NumiHumanRegisteredLigaments.hpp').read_bytes())
    skin=src/'NumiHumanLoadedSkinPose.hpp';(model/skin.name).unlink();(model/skin.name).write_bytes(skin.read_bytes())
    (model/main).write_bytes((src/main).read_bytes())
    cfg=json.loads((baseline/'launch.json').read_bytes());actual=json.loads((fk/'target-pose.json').read_bytes());p=[x['current'] for x in actual['all_bodies']]
    current=np.array([x['position_world_m']+x['orientation_world_xyzw'] for x in p],'f4')
    previous=read_pose(h.parent/'native-ligament-material/stderr.log');base=np.array(previous['bodyPoses'],'f4')
    source=Path(cfg['argv'][3]);bone=banks(source);cortices={}
    if previous['bonePayloadSha256']!=digest(source) or len(bone)!=185:raise ValueError('Changed source bank')
    geometry=[]
    for f in sorted((h/'oriented-routes').glob('*.json')):
        row=json.loads(f.read_bytes());s=json.loads((h/'section-routes'/f.name).read_bytes());pts=np.array(row['world_m'])*1000.
        for body in {e['body'] for e in row['edges']}|{e['body'] for e in s['sites']}:
            if not np.array_equal(current[body],base[body]):raise ValueError('Cortical force owner moved relative to the validated paths')
        near=[]
        for (b,sid),(face,v,raw) in bone.items():
            pos=np.array(current[b],float);w=Rotation.from_quat(pos[3:]).apply(v)*1000+pos[:3]*1000
            if np.all(w.max(0)+.001>=pts.min(0)) and np.all(w.min(0)-.001<=pts.max(0)):
                near.append(sid)
                # Source registration uses native positions as serialized at max_digits10;
                # those represent the same Float32 values. Retain their original
                # double conversion for bones with bit-identical native frames.
                pp=previous['bodyPoses'][b] if np.array_equal(base[b],current[b]) else p[b]['position_world_m']+p[b]['orientation_world_xyzw']
                if sid not in cortices:cortices[sid]=Cortex(b,sid,(face,v,raw),pp)
        defects=verify(pts,[cortices[k] for k in near])
        if defects:raise ValueError(f'Current reference path intersects {defects}')
        geometry.append(dict(name=row['fiber'],broadphase_banks=near,crossings=defects))
    s=(src/reg).read_text();a=s.index('inline constexpr float registeredCorticalPose[][7]={')
    s=s[:a]+'inline constexpr float registeredCorticalPose[][7]={\n'+',\n'.join('{'+','.join(map(fl,rr))+'}' for rr in current)+'\n};\n';(model/reg).write_text(s)
    rr=h/'closed-native/native';x=np.loadtxt(rr/'mesh.node',skiprows=1,usecols=(1,2,3));y=x.copy();t=np.loadtxt(rr/'mesh.ele',skiprows=1,dtype='i4',usecols=(1,2,3,4))-1
    de=np.load(field/'material-transport.npz');y[de['nodes']]=de['positions']
    v,L=measure(y,t);v0,_=measure(x,t);relative=v/v0
    failures=dict(inverted=int((v<=0).sum()),volume=int((v<=6e-18).sum()),shape=int((v<L*1.e-5).sum()),compressed=int((relative<.4).sum()),expanded=int((relative>2.5).sum()))
    if any(failures.values()):raise ValueError(f'Loaded material rejected {failures}')
    bound=json.loads((h/'scene-registration/registration.json').read_bytes());pinned=bound['first_contact_new']+[r[2] for r in bound['anchors']]
    if not np.array_equal(x[pinned],y[pinned]):raise ValueError('Registered attachment or bed witness moved')
    path=out/'current.node'
    with path.open('x') as f:
        f.write(str(len(y))+' 3 0 0\n');np.savetxt(f,np.c_[np.arange(len(y))+1,y],fmt=['%d','%.17g','%.17g','%.17g'])
    s=(src/unit).read_text();s=exact(s,'char nodeSHA[]="'+digest(rr/'mesh.node')+'"','char nodeSHA[]="'+digest(path)+'"')
    pattern=r'(constexpr Body target\[\]=\{)(.*?)(\};)';block=re.search(pattern,s,re.S);ids=[int(a) for a in re.findall(r'^\{(\d+),',block[2],re.M)]
    points='\n'+''.join('{'+str(i)+',{'+','.join(map(fl,current[i,:3]))+'},{'+','.join(map(fl,current[i,3:]))+'}},\n' for i in ids)
    (model/unit).write_text(s[:block.start(2)]+points+s[block.end(2):])
    data=json.loads((build/'build.json').read_bytes());commands=[[a.replace(str(src)+'/',str(model)+'/') for a in c] for c in data['commands'][-2:]]
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_';env=cfg['environment']
    if digest(rr/'mesh.node')!=env[prefix+'TETGEN_NODE_SHA256']:raise ValueError('Different material rest owner')
    with (out/'build.log').open('wb') as log:
        for c in commands:subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Select this input posture from the actually executed native FK trial.
    kcfg=json.loads((fk/'launch.json').read_bytes());argv=kcfg['argv'];job=out/'scene';job.mkdir()
    argv[0]=str(model/'numi-human-native');argv[4]=str(job/'native-traces');env={k:v.replace(str(baseline)+'/',str(job)+'/') for k,v in env.items()}
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(path);cfg['argv']=argv;cfg['environment']=env;cfg['pins'][str(path)]=digest(path);cfg['pins'][argv[0]]=digest(Path(argv[0]))
    cfg['pose_fit']=digest(fk/'target-pose.json');cfg['parent_launch']=digest(baseline/'launch.json')
    (job/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    report=dict(source_pose=digest(fk/'target-pose.json'),material_rest=digest(rr/'mesh.node'),material_current=digest(path),elements=digest(rr/'mesh.ele'),cell_mass=digest(rr/'cell-mass.f64'),node_mass=digest(rr/'node-mass.f64'),shape=failures,minimum_shape=float(np.min(v/L)),relative_volume_range=[float(relative.min()),float(relative.max())],bed_nodes=bound['first_contact_new'],deep_anchors_unchanged=len(bound['anchors']),curves=geometry,app=digest(Path(argv[0])),source_binding=digest(model/unit),commands=commands)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(execute_native(argv,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,job,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
