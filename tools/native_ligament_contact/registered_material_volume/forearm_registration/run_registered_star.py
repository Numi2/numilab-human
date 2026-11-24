"""Build and replay the same full Metal scene with a registered internal starting pose."""
import json,sys,os,shutil,subprocess
from pathlib import Path
import numpy as np
from build_bed_audit import sha,replace_once
from run_cavity_motion import execute_native
def run(source,binary,contact,fit,out,steps):
    cfg=json.loads((source/'launch.json').read_bytes());env=cfg['environment'];pins=cfg['pins'];argv=cfg['argv'];f=json.loads((fit/'fit.json').read_bytes())
    key='NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE';old=Path(env[key])
    if sha(old)!=f['control_loaded'] or sha(fit/'current.node')!=f['registered']:raise ValueError('Unmatched registration')
    interval=json.loads((fit/'interval.json').read_bytes());walls=json.loads((fit/'wall-crossings.json').read_bytes())
    if interval['input']!=f['registered'] or interval['failed'] or np.sum(walls['pair_counts']):raise ValueError('Inadmissible geometry')
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='1':raise ValueError('Incomplete scene')
    z=np.load(contact/'material-contact.npz');nodes=f['nodes_0_based'];r=json.loads((contact/'result.json').read_bytes())
    if r['loaded']!=f['control_loaded'] or r['program']!=sha(contact/'material-contact.npz') or np.isin(z['facets'][:,4:7],nodes).any() or np.isin(z['samples'][:,0],nodes).any():raise ValueError('Cortical contact geometry changed')
    spec=json.loads((binary/'build.json').read_bytes());build=fit/'binary'
    if sha(Path(argv[0]))!=spec['executable']:raise ValueError('Unexpected source executable')
    if not (build/'build.json').exists():
        build.mkdir(exist_ok=True)
        for p in binary.iterdir():
            if p.suffix not in ('.hpp','.inc','.mm','.h'):continue
            if p.name in ('NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc'):
                (build/p.name).write_text(replace_once(p.read_text(),f['control_loaded'],f['registered']))
            else:shutil.copyfile(p,build/p.name)
        cmds=[[s.replace(str(binary)+'/',str(build)+'/') for s in cmd] for cmd in spec['commands']]
        for i,cmd in enumerate(cmds):
            with (build/f'build-{i}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        (build/'build.json').write_text(json.dumps(dict(commands=cmds,executable=sha(build/'numi-human-native'),source=sha(binary/'build.json')),indent=2)+'\n')
    if steps==0:return
    for v in set(argv+list(env.values())).intersection(pins):
        if sha(Path(v))!=pins[v]:raise ValueError('Stale input '+v)
    env[key]=str(fit/'current.node');pins[env[key]]=f['registered'];argv[0]=str(build/'numi-human-native');pins[argv[0]]=sha(Path(argv[0]))
    argv[4]=str(out/'native-traces');argv[argv.index('--muscle-step-count')+1]=str(steps)
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps);env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']=str(steps)
    for k,val in list(env.items()):
        if val.startswith(str(source)+'/') and val not in pins:env[k]=str(out/Path(val).relative_to(source))
    out.mkdir();cfg.update(parent_launch=sha(source/'launch.json'),forearm_registration=sha(fit/'fit.json'))
    (out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    passed={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}
    result=execute_native(argv,cfg['cwd'],passed|env,out,minimum_free_bytes=4*2**30,timeout_s=650)
    print(result,flush=True)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:-1]),int(sys.argv[-1]))
