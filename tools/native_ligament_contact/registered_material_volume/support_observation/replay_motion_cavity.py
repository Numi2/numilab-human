"""Launch the complete coupled scene after geometry, material, cortical and mass admission."""
from pathlib import Path
import json,os,sys
from build_bed_audit import sha
from run_cavity_motion import execute_native
def run(control,stage,binding,out,steps):
    read=lambda p:json.loads(p.read_bytes());cfg=read(control/'launch.json');env=cfg['environment'];argv=cfg['argv'];pins=cfg['pins'];native=read(binding/'build.json');vol=stage/'volume'
    if steps not in (1,4) or env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1' or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='1':raise ValueError('Invalid physical comparator')
    changed=native['tables'].copy()
    for f in ('material-p1/result.json','packets/result.json','native-replay/result.json'):
        r=read(stage/f)
        if r.get('exit',0) or any(v.get('inadmissible_label_rows') for v in r.get('roles',{}).values()):raise ValueError('Source mass or basis admission failed')
        changed.update(r['environment'])
    pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for k,f in [('TETGEN_NODE','mesh.node'),('TETGEN_ELEMENTS','mesh.ele'),('CHILD_LABELS','current-child.labels.csv')]:
        changed[pre+k]=str(vol/f);changed[pre+k+'_SHA256']=sha(vol/f)
    check=read(vol/'native-import.json')
    if check['label_sha256']!=read(vol/'result.json')['inputs']['current-child.labels.csv'] or check['element_sha256']!=native['topology']:raise ValueError('Different anatomical registration')
    bypath={};bysha={}
    for k,val in changed.items():
        old=env.get(k)
        if old in pins and val.startswith('/') and Path(val).is_file():
            bypath[old]=val;h=sha(Path(val));oldhash=pins[old]
            if oldhash in bysha and bysha[oldhash]!=h:raise ValueError('Conflicting matched sources for '+k)
            bysha[oldhash]=h;pins[val]=h
    for k,v in list(env.items()):env[k]=bypath.get(v,bysha.get(v,v))
    env.update(changed)
    argv[0]=str(binding/'app/numi-human-native');pins[argv[0]]=native['executable'];argv[4]=str(out/'native-traces')
    argv[argv.index('--muscle-step-count')+1]=str(steps)
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps);env['NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP']=str(steps)
    for k,val in list(env.items()):
        if val.startswith(str(control)+'/') and val not in pins:env[k]=str(out/Path(val).relative_to(control))
    env['DYLD_LIBRARY_PATH']=str(binding/'runtime')+':'+env['DYLD_LIBRARY_PATH'];lib=binding/'runtime/libmetalrobo.dylib'
    if sha(lib)!=native['library']:raise ValueError('Wrong mechanics library')
    pins[str(lib)]=native['library'];cfg['parent_launch']=sha(control/'launch.json');cfg['cell_transfer']=sha(vol/'transfer.json');cfg['native_carrier_replay']=sha(stage/'native-replay/result.json')
    for f in set(argv+list(env.values())) & set(pins):
        if sha(Path(f))!=pins[f]:raise ValueError('Stale input '+f)
    out.mkdir();(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    code=execute_native(argv,cfg['cwd'],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=650)
    print(code,flush=True)
if __name__=='__main__':run(*(Path(x).resolve()for x in sys.argv[1:-1]),int(sys.argv[-1]))
