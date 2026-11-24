"""Advance the exact integrated native reference with the newly registered interior coordinates."""
import argparse,json,os,shutil
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(source,stage,out,steps):
    src=source.resolve(); s=stage/'witness-material';v=s/'volume'; owner=s/'binding';read=lambda p:json.loads(p.read_bytes())
    native=read(owner/'build.json');nativevol=read(v/'result.json'); cfg=read(src/'launch.json')
    env=cfg['environment']; argv=cfg['argv']; pins=cfg['pins']
    if steps not in (1,4) or not nativevol['native']['native_imported']:raise ValueError('Unsupported matched comparison')
    if env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='1':raise ValueError('Incomplete variable-pressure body')
    changes={}
    for f in ('current-p1/result.json','recorded-mass/packets/result.json','recorded-mass/native-replay/result.json','binding/tables/result.json'):
        r=read(s/f)
        if r.get('exit',0) or any(role['inadmissible_label_rows'] for role in r.get('roles',{}).values()):
            raise ValueError('Source allocation gate failed')
        changes.update(r['environment'])
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for k,n in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele'),('CHILD_LABELS','current-child.labels.csv')]:
        changes[pre+k]=str(v/n);changes[pre+k+'_SHA256']=digest(v/n)
    changes['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(v/'current.node')
    if digest(v/'current.node')!=native['loaded'] or digest(owner/'app/numi-human-native')!=native['app']:raise ValueError('Wrong native binding')
    substitutions={}; hashes={}
    for k,new in changes.items():
        old=env.get(k)
        if old in pins and new.startswith('/') and Path(new).is_file():
            if old in substitutions and substitutions[old]!=new:raise ValueError('Conflicting paths')
            substitutions[old]=new
            a,b=pins[old],digest(Path(new))
            if a in hashes and hashes[a]!=b:raise ValueError('Conflicting identities')
            hashes[a]=b
    for k,old in list(env.items()):
        env[k]=substitutions.get(old,hashes.get(old,old))
    for old,new in substitutions.items():pins[new]=hashes[pins[old]]
    env.update(changes)
    argv[0]=str(owner/'app/numi-human-native');pins[argv[0]]=native['app']
    argv[argv.index('--muscle-step-count')+1]=str(steps)
    argv[4]=str(out/'native-traces')
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
    env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']=str(steps)
    for k,val in list(env.items()):
        if val.startswith(str(src)+'/') and val not in pins:env[k]=str(out/Path(val).relative_to(src))
    for val in set(argv+list(env.values()))&set(pins):
        if digest(Path(val))!=pins[val]:raise ValueError('Stale runtime input '+val)
    lib=next(Path(d)/'libmetalrobo.dylib' for d in env['DYLD_LIBRARY_PATH'].split(':') if (Path(d)/'libmetalrobo.dylib').exists())
    if digest(lib)!=native['library']:raise ValueError('Unmatched physical library')
    reserve=shutil.disk_usage(out.parent).free
    cfg.update(parent_launch=digest(src/'launch.json'),free_bytes_before=reserve,material_registration=digest(owner/'tables/result.json'),carrier_replay=digest(s/'recorded-mass/native-replay/result.json'))
    out.mkdir(exist_ok=False);(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=600),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('source','stage','out'):p.add_argument(a,type=lambda x:Path(x).resolve())
    p.add_argument('steps',type=int);run(**vars(p.parse_args()))
