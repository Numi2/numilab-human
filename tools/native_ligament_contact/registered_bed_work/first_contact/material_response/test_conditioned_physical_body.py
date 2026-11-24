"""Verify the coupled registration, rebuild both native owners, and run the first accepted-state transaction."""
from pathlib import Path
import json,os,sys,shutil,subprocess
import numpy as np
from audit_loaded_registration_fit import run as audit
from measure_bone_tissue_overlap import run as bone_check
from compile_connected_joint import digest
from run_cavity_motion import execute_native
from register_loaded_cortical_interfaces import run as cortical
from compile_cortical_skin_contact import run as quadrature
from build_cortical_skin_registration import write_contact_include

def run(job,h,trial,interval,layerlib,cortexlib):
    read=lambda p:json.loads(p.read_bytes())
    qual=job/'interval-certified-scene';mat=qual/'material';shader=job/'persistent-volume-witness';runtime=job/'certified-heart-runtime'
    audit(job,trial,interval,layerlib)
    if read(trial/'interval-audit.json')['failed_intervals'] or np.any(read(trial/'loaded-layers.json')['pair_counts']):raise ValueError('Candidate failed interval or tissue layers')
    x,y=[np.loadtxt(mat/p,skiprows=1,usecols=(1,2,3)).astype('f4') for p in ['mesh.node','current.node']]
    sel=np.load(trial/'candidate-positions.npz');ids=sel['nodes'];y[ids]=sel['positions'];changed=np.flatnonzero(np.any(x!=y,axis=1))
    np.savez_compressed(trial/'actual-transport.npz',nodes=changed,positions=y[changed])
    bone_check(h,trial/'bone-overlap',job/'distal-registered-native/target-pose.json',trial/'actual-transport.npz',mat)
    prior=read(qual/'bone-audit/result.json')['results'];now=read(trial/'bone-overlap/result.json')['results']
    if set(prior)!=set(now) or len(now)!=185 or any(now[k][role]>prior[k][role]+1e-6 for k in prior for role in ['skin','fat']):raise ValueError('New cortical overlap')
    p=trial/'current.node'
    with (mat/'current.node').open() as old,p.open('x') as new:
        new.write(next(old));changed=set(ids.tolist())
        for i,line in enumerate(old):
            new.write(str(i+1)+' '+' '.join(format(float(c),'.17g') for c in y[i])+'\n' if i in changed else line)
    cortical(h,job,cortexlib,trial/'cortex',p)
    quadrature(h,job,job/'physical-weight-frames',trial/'contact',mat,p,trial/'cortex')
    for key in ['cortex','contact']:
        name='registered-cortical-facets.npz' if key=='cortex' else 'material-contact.npz'
        a=np.load(qual/key/name);b=np.load(trial/key/name)
        if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Transferred cortical owners changed')
    app=trial/'app';app.mkdir();src=runtime/'app';lp='NumiHumanLoadedSkinPoseData.inc';cp='NumiHumanCorticalSkinBinding.inc'
    for f in src.iterdir():
        if f.name not in (lp,cp) and f.suffix in ('.h','.hpp','.mm','.inc'):os.link(f,app/f.name)
    before=digest(mat/'current.node');after=digest(p);content=(src/lp).read_text()
    if content.count(before)!=1:raise ValueError('Material owner mismatch')
    (app/lp).write_text(content.replace(before,after));write_contact_include(trial/'contact',app)
    if (app/cp).read_text()!=(src/cp).read_text().replace(before,after):
        # Contact provenance can change even when every face, area and law remains identical.
        a=(app/cp).read_text().splitlines();b=(src/cp).read_text().splitlines()
        if len(a)!=len(b) or any(x!=y and not(x.startswith('inline constexpr char ') and y.startswith('inline constexpr char ') and x.partition('=')[0]==y.partition('=')[0]) for x,y in zip(a,b)):raise ValueError('Contact law or identities changed')
    cmds=[[s.replace(str(src)+'/',str(app)+'/') for s in cmd] for cmd in read(runtime/'build.json')['commands']]
    with (trial/'build.log').open('w') as f:
        for cmd in cmds:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    scene=trial/'scene';scene.mkdir();old=runtime/'scene';cfg=read(old/'launch.json')
    cfg['parent_launch']=digest(old/'launch.json');exe=app/'numi-human-native';lib=shader/'metal/NumiMatter.metallib';cfg['argv'][0]=str(exe);cfg['argv'][4]=str(scene/'native-traces')
    env=cfg['environment']
    for k,v in list(env.items()):
        if v.startswith(str(old)+'/') and v not in cfg['pins']:env[k]=str(scene/Path(v).relative_to(old))
    env['NUMI_MATTER_METALLIB_OVERRIDE']=str(lib);env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(p)
    for f in [exe,p,lib]:cfg['pins'][str(f)]=digest(f)
    for v in cfg['argv']+list(env.values()):
        if v in cfg['pins'] and digest(Path(v))!=cfg['pins'][v]:raise ValueError('Changed native input')
    (trial/'build.json').write_text(json.dumps(dict(commands=cmds,model=digest(exe),shader=digest(lib),reference=digest(mat/'mesh.node'),loaded=digest(p),topology=digest(mat/'mesh.ele'),cell_mass=digest(mat/'cell-mass.f64'),node_mass=digest(mat/'node-mass.f64')),indent=2)+'\n')
    (scene/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,scene,minimum_free_bytes=4*2**30,timeout_s=560),flush=True)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
