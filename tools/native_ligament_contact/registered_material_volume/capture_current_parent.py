"""Export the compiled native parent from the same selected material input and release pose."""
from pathlib import Path
import json,os,subprocess,sys,shutil
from emit_skin_source_registration import sha
src,out,app,template,selection,seed,trial=map(Path,sys.argv[1:]);out.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.hpp','.metalinc','.metal') and not (out/f.name).exists():(out/f.name).hardlink_to(f)
(out/'metalrobo').mkdir(exist_ok=True)
for f in (src/'metalrobo').glob('*.metallib'):
    if not (out/'metalrobo'/f.name).exists():(out/'metalrobo'/f.name).hardlink_to(f)
n='numilab_human_myosim_visual_probe.mm';s=(src/n).read_text()
def change(a,b):
    global s
    if s.count(a)!=1:raise ValueError('Unknown source '+a)
    s=s.replace(a,b)
a='''if(!full||std::strcmp(full,"1")!=0){error="selected physical skin requires its full source volume";return false;}'''
test='''const char* capture=std::getenv("NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY");
        const bool parentCapture=capture&&std::strcmp(capture,"1")==0
            &&std::getenv("NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV");'''
change(a,test+'\n        '+a.replace('if(!full||std::strcmp(full,"1")!=0)','if(!parentCapture&&(!full||std::strcmp(full,"1")!=0))'))
change('if(selectedParent)require(registeredVolumeInstallRequested&&preparedRegisteredVolumeBundle.has_value(),',test+'''
        const bool exportSelectedParent=selectedParent&&parentCapture&&preparationOnly&&!registeredVolumeInstallRequested;
        if(selectedParent)require(exportSelectedParent||(registeredVolumeInstallRequested&&preparedRegisteredVolumeBundle.has_value()),''')
change('require(kWholeBodySkinLayersAdmitted||exactUnadmittedDiagnostic||replacementParentPreparationOnly,',
       'require(kWholeBodySkinLayersAdmitted||exactUnadmittedDiagnostic||replacementParentPreparationOnly||exportSelectedParent,')
(out/n).write_text(s);cmd=[[a.replace(str(src)+'/',str(out)+'/') for a in row] for row in json.loads((src/'build-commands.json').read_text())]
(out/'build-commands.json').write_text(json.dumps(cmd,indent=2)+'\n')
if os.environ.get('NUMI_REPLAY_COMPILED_PARENT')=='1':
    check=json.loads((out/'identity.json').read_text())
    for path,h in check.items():
        if sha(Path(path))!=h:raise ValueError('Compiled capture owner drift '+path)
else:
    for row in cmd:subprocess.run(row,cwd=app/'build-native',check=True)
    (out/'identity.json').write_text(json.dumps({str(p):sha(p) for p in [src/n,out/n,out/'numi-human-native',*(out/'metalrobo').glob('*.metallib')]},indent=2)+'\n')
a,b,c=[json.loads(p.read_text()) for p in (template,selection,seed)]
replaced={a['argv'][0],*[a['argv'][a['argv'].index(flag)+1] for flag in ('--skin-payload','--soft-tissue-payload','--torso-anatomy-payload','--resting-anatomy-receipt')]}
for p,h in a['pins'].items():
    if p in replaced:continue
    if sha(Path(p))!=h:raise ValueError('Input drift '+p)
trial.mkdir(exist_ok=False)
argv=a['argv'].copy();argv[0]=str(out/'numi-human-native');argv[4]=str(trial/'native')
for flag in ('--skin-payload','--soft-tissue-payload','--torso-anatomy-payload','--resting-anatomy-receipt'):
    argv[argv.index(flag)+1]=b['argv'][b['argv'].index(flag)+1]
i,j=argv.index('--root-pose')+1,c['argv'].index('--root-pose')+1
argv[i:i+7]=c['argv'][j:j+7]
env={k:v.replace(str(template.parent),str(trial)) for k,v in a['environment'].items() if 'EXPERIMENTAL_UNADMITTED_LAYER_' not in k}
assert env['NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY']=='1' and 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_INSTALL' not in env
for k,v in c['environment'].items():
    if k.endswith(('SURFACE_O_NODES_SHA256','SURFACE_O_NODES_CSV')):env[k]=v
receipt=dict(argv=argv,environment=env,cwd=a['cwd'],pins={str(p):sha(p) for p in [template,selection,seed,out/'numi-human-native']})
(trial/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (trial/'stdout.log').open('w') as o,(trial/'stderr.log').open('w') as e:
    result=subprocess.run(argv,cwd=a['cwd'],env={**os.environ,**env},stdout=o,stderr=e,timeout=400)
log=(trial/'stdout.log').read_text()
if result.returncode or 'skin_fat_parent_mass_capture_only=PASS physical_steps=0 runtime_created=0' not in log:raise RuntimeError((trial/'stderr.log').read_text()[-650:])
(trial/'receipt.json').write_text(json.dumps(dict(outputs={x.name:sha(x) for x in trial.glob('compiled-parent-cell-mass.csv*')},result=[l for l in log.splitlines() if l.startswith(('whole_body_compiled_skin_fat_parent_mass=','skin_fat_parent_mass_capture_only='))]),indent=2)+'\n')
