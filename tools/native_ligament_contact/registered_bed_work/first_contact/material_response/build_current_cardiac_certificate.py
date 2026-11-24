"""Admit the verified cardiac subset to the existing pressure-dependent geometry certificate."""
from pathlib import Path
import json,sys,subprocess,os
from compile_connected_joint import digest
from make_shared_tissue_contact import substitute
from run_cavity_motion import execute_native

def run(job,source,solver,base,out):
    read=lambda p:json.loads(p.read_bytes());proof=read(job/'current-cardiac-records.json');ranges=proof['ranges']
    if len(ranges)!=152 or sum(r['vertices'] for r in ranges)!=621369 or any(r['changed_vertices'] or not r['triangle_parity'] for r in ranges):raise ValueError('Unqualified cardiac field')
    cfg=read(base/'launch.json');a=cfg['argv'];nha=Path(a[a.index('--torso-anatomy-payload')+1]);receipt=Path(a[a.index('--resting-anatomy-receipt')+1])
    if proof['current']!=digest(nha) or digest(nha)!=cfg['pins'][str(nha)]:raise ValueError('Wrong current geometry')
    c=read(receipt)['provenance']['cardiac_geometry_binding']['common_field']
    for k,v in proof['sidecars'].items():
        if digest(receipt.parent/c[k]['path'])!=v['sha256']:raise ValueError('Mismatched cardiac map')
    h='NumiHumanRestingVisual.hpp';s=(source/'app'/h).read_text();out.mkdir();app=out/'app';app.mkdir()
    for f in (source/'app').iterdir():
        if f.name!=h and f.suffix in ('.inc','.hpp','.mm','.h'):os.link(f,app/f.name)
    check='functional.commonFieldAnatomyPayloadSHA256=="'+proof['original']+'"'
    s=substitute(s,check,'('+check+'||functional.commonFieldAnatomyPayloadSHA256=="'+proof['current']+'")')
    (app/h).write_text(s)
    native=read(source/'build.json')['commands'][4];link=read(solver/'build.json')['commands'][-1]
    native=[x.replace(str(source/'app')+'/',str(app)+'/') for x in native];link=[x.replace(str(source/'app/numi-human-native.o'),str(app/'numi-human-native.o')).replace(str(solver/'app/numi-human-native'),str(app/'numi-human-native')) for x in link]
    with (out/'build.log').open('w') as log:
        for cmd in [native,link]:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=app/'numi-human-native'
    if str(solver/'libmetalrobo.dylib') not in subprocess.check_output(['otool','-L',str(exe)],text=True):raise ValueError('Runtime library mismatch')
    (out/'build.json').write_text(json.dumps(dict(binary=digest(exe),commands=[native,link],header=digest(app/h),proof=digest(job/'current-cardiac-records.json'),original=proof['original'],current=proof['current'],map=proof['sidecars']['map']['sha256']),indent=2)+'\n')
    scene=out/'scene';scene.mkdir();a[0]=str(exe);a[4]=str(scene/'native-traces');cfg['pins'][str(exe)]=digest(exe);cfg['parent_launch']=digest(base/'launch.json');env={k:v.replace(str(base)+'/',str(scene)+'/') if k.endswith(('_OUTPUT','_OUTPUT_PATH','_CSV_OUTPUT','_DUMP')) else v for k,v in cfg['environment'].items()}
    # The parent's own authored assets stay at their verified input paths; redirect
    # only native capture destinations that previous launch code put inside base.
    for k,v in cfg['environment'].items():
        if v.startswith(str(base)+'/') and (v in cfg['pins'] or Path(v).name in {'resting-anatomy-receipt.json','bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'}):env[k]=v
        elif v.startswith(str(base)+'/'):env[k]=v.replace(str(base)+'/',str(scene)+'/',1)
    cfg['environment']=env;(scene/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(a,cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,scene,minimum_free_bytes=4*2**30,timeout_s=400),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
