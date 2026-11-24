"""Evaluate the existing native bed owner with the tangent of the registered material law."""
import json,os,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def build(source,params,dest):
    dest.mkdir(exist_ok=True);name='NumiHumanBedColumnQuadrature.hpp'
    parent=json.loads((source/'identities.json').read_bytes())
    if subprocess.check_output(['git','hash-object',str(source/name)],text=True).strip()!='bc14eed3c92bf6e942879ebdddba3746a75d53eb':raise ValueError('Mismatched bed owner at e9d5d16d')
    mat=json.loads(params.read_bytes())['materials'];text=(source/name).read_text()
    values={}
    for m,role,old in [('skin','15','200000.0'),('fat','16','55000.0')]:
        p=mat[role]['registered'];mu=p['source_mu_pa'];lam=p['source_logJ_coefficient_pa'];E=lam+2*mu;values[m]=E
        a=f'double {m}ModulusPa = {old};'
        if text.count(a)!=1:raise ValueError('Changed column material')
        text=text.replace(a,f'double {m}ModulusPa = {E.hex()};')
    (dest/name).write_text(text)
    for f in source.iterdir():
        if f.name==name or f.name.startswith('build') or f.name=='numi-human-native' or f.suffix=='.o':continue
        target=dest/f.name
        if not target.exists():
            # The Objective-C++ translation unit resolves quoted headers beside itself.
            if f.suffix=='.mm':target.write_bytes(f.read_bytes())
            else:target.symlink_to(f.resolve(),target_is_directory=f.is_dir())
    commands=[[s.replace(str(source)+'/',str(dest)+'/')for s in cmd]for cmd in json.loads((source/'build-commands.json').read_bytes())[2:]]
    control=json.loads((source.parent/'motion-16/launch.json').read_bytes())
    lib=Path(control['environment']['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib'
    if digest(lib)!=control['pins'][str(lib)]:raise ValueError('Changed native runtime')
    with (dest/'compile.log').open('w')as log:
        for cmd in commands:subprocess.run(cmd,cwd=lib.parent.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    (dest/'registered-sections.json').write_text(json.dumps(dict(material=digest(params),header=digest(dest/name),source_header=digest(source/name),original_binary=digest(source/'numi-human-native'),binary=digest(dest/'numi-human-native'),confined_tangent_pa=values,commands=commands),indent=2)+'\n')

def trial(reference,build,case,steps):
    cfg=json.loads((reference/'launch.json').read_bytes());argv=cfg['argv'];env=cfg['environment'];pins=cfg['pins']
    if env['NUMI_HUMAN_LOCAL_BED_LAYERS']!='1' or env['NUMI_HUMAN_LOCAL_BED_REGIONAL_THICKNESS']!='1' or env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Wrong physical operating mode')
    if cfg['headless'] or env['NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM']!='0':raise ValueError('Expected existing reduced native control')
    for val in set(argv+list(env.values()))&set(pins):
        if (Path(val).is_file()and digest(Path(val))!=pins[val])or not Path(val).exists():raise ValueError('Changed experiment input '+Path(val).name)
    case.mkdir();argv[0]=str(build/'numi-human-native');argv[4]=str(case/'native-traces');pins[argv[0]]=digest(Path(argv[0]));argv[argv.index('--muscle-step-count')+1]=str(steps)
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
    for k,val in list(env.items()):
        if val.startswith(str(reference)+'/') and val not in pins:env[k]=str(case/Path(val).relative_to(reference))
    for name in ('MetalRobo.metallib','HumanRestingSkin.metallib'):
        gpu=build/'metalrobo'/name
        pins[str(gpu)]=digest(gpu)
    if (build/'curves.json').exists():
        manifest=json.loads((build/'curves.json').read_bytes())
        if manifest['binary']!=pins[argv[0]] or manifest['gpu']!=pins[str(build/'metalrobo/MetalRobo.metallib')]:
            raise ValueError('Wrong registered finite-strain bundle')
    cfg['matched_reference_launch']=digest(reference/'launch.json')
    (case/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,case,minimum_free_bytes=4*2**30,timeout_s=900),flush=True)
if __name__=='__main__':
    if sys.argv[1]=='build':build(*(Path(s).resolve()for s in sys.argv[2:]))
    else:trial(*(Path(s).resolve()for s in sys.argv[2:5]),int(sys.argv[5]))
