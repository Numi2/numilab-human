"""Install a registered starting pose with unchanged material, anatomical anchors and real contacts."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include
from run_cavity_motion import execute_native

def run(source,fit,control,output,steps):
    v=source/'volume';owner=source/'binding'
    data=json.loads((fit/'fit.json').read_bytes())
    for f,key in [('mesh.node','source_reference'),('current.node','control_loaded'),('mesh.ele','elements'),('node-mass.f64','source_mass')]:
        if digest(v/f)!=data[key]:raise ValueError('Changed '+f)
    if data['registered']!=digest(fit/'current.node'):raise ValueError('Unpinned pose')
    interval=json.loads((fit/'interval.json').read_bytes())
    if interval['input']!=data['registered'] or interval['failed']:raise ValueError('Invalid determinant interval')
    walls=json.loads((fit/'wall-crossings.json').read_bytes())
    if sum(map(sum,walls['pair_counts'])):raise ValueError('Invalid material surfaces')
    build=fit/'binding';app=build/'app';con=build/'contact'
    if not (build/'build.json').exists():
        for d in (app,con):d.mkdir(exist_ok=True,parents=True)
        source_app=owner/'app'
        for f in source_app.iterdir():
            if f.suffix in ('.h','.hpp','.mm','.inc') and f.name not in ('NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc'):
                subprocess.run(['/bin/cp','-cp',f,app/f.name],check=True)
        n='NumiHumanLoadedSkinPoseData.inc'
        code=(source_app/n).read_text()
        if code.count(data['control_loaded'])!=1:raise ValueError('Unexpected pose pin')
        (app/n).write_text(code.replace(data['control_loaded'],data['registered']))
        contact=json.loads((owner/'contact/result.json').read_bytes())
        if contact['loaded']!=data['control_loaded']:raise ValueError('Mismatched cortical contact')
        contact['loaded']=data['registered']
        (con/'result.json').write_text(json.dumps(contact,indent=2)+'\n')
        program=con/'material-contact.npz'
        if not program.exists():program.symlink_to((owner/'contact'/program.name).resolve())
        write_contact_include(con,app)
        cmds=[]
        ident=json.loads((owner/'build.json').read_bytes())
        for prior in ident['commands']:
            cmd=[a.replace(str(source_app)+'/',str(app)+'/') if a!=str(source_app/'app-runtime.mm.o') else a for a in prior]
            cmds.append(cmd)
            with (build/f'compile-{len(cmds)}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        (build/'build.json').write_text(json.dumps(dict(commands=cmds,app=digest(app/'numi-human-native'),library=ident['library'],loaded=data['registered']),indent=2)+'\n')
    cfg=json.loads((control/'launch.json').read_bytes());env=cfg['environment'];argv=cfg['argv']
    if Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']).resolve()!=(v/'current.node').resolve() or env['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise ValueError('Mismatched control')
    env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']=str(fit/'current.node')
    cfg['pins'][str(fit/'current.node')]=data['registered']
    for k,val in list(env.items()):
        if val.startswith(str(control)+'/') and val not in cfg['pins']:env[k]=str(output/Path(val).relative_to(control))
    argv[0]=str(app/'numi-human-native');argv[4]=str(output/'native-traces')
    argv[argv.index('--muscle-step-count')+1]=str(steps)
    cfg['pins'][argv[0]]=json.loads((build/'build.json').read_bytes())['app']
    env['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']=str(steps)
    for val in set(argv+list(env.values())) & cfg['pins'].keys():
        if digest(Path(val))!=cfg['pins'][val]:raise ValueError('Stale '+val)
    output.mkdir()
    cfg.update(parent_launch=digest(control/'launch.json'),registration_fit=digest(fit/'fit.json'),free_bytes_before=shutil.disk_usage(output).free)
    (output/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}
    print(execute_native(argv,cfg['cwd'],clean|env,output,minimum_free_bytes=4*2**30,timeout_s=650),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('source','fit','control','output'):p.add_argument(n,type=lambda x:Path(x).resolve())
    p.add_argument('steps',type=int)
    run(**vars(p.parse_args()))
