"""Recover the original source-to-bed rigid transform from the native producer that supplied the mass ledger."""
import os,json,sys,subprocess
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(source,control,out):
    out.mkdir();app=out/'app';app.mkdir();original=source/'numilab_human_myosim_visual_probe.mm'
    for f in source.iterdir():
        if f.name!=original.name and f.suffix in ('.hpp','.mm','.h','.inc','.metalinc'):os.link(f.resolve(),app/f.name)
    s=original.read_text();needle='            for(auto& p:region.worldPoints)p=skinPlacement.point(p);'
    if s.count(needle)!=1:raise ValueError('Cannot locate the tested source placement')
    s=s.replace(needle,'''            std::ostringstream sourcePlacement;sourcePlacement<<std::hexfloat;
            for(double value:skinPlacement.rotation)sourcePlacement<<value<<" ";
            for(double value:skinPlacement.translation)sourcePlacement<<value<<" ";
            std::cout<<"captured_coarse_material_placement="<<sourcePlacement.str()<<std::endl;
            throw std::runtime_error("source placement capture complete before material installation");
'''+needle);(app/original.name).write_text(s)
    build=json.loads((source/'build.json').read_bytes());cfg=json.loads((control/'launch.json').read_bytes())
    if digest(Path(cfg['argv'][0]))!=build['executable']['sha256']:raise ValueError('Control executable does not match producer build')
    cmds=[[v.replace(str(source)+'/',str(app)+'/') for v in c] for c in build['commands']]
    with (out/'build.log').open('w') as f:
        for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
    exe=app/'numi-human-native';new=out/'scene';new.mkdir();cfg['argv'][0]=str(exe);cfg['argv'][4]=str(new/'native-traces');env={k:v.replace(str(control)+'/',str(new)+'/') for k,v in cfg['environment'].items()};cfg['environment']=env;cfg['pins'][str(exe)]=digest(exe)
    (new/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    run=execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,new,minimum_free_bytes=4*2**30,timeout_s=400)
    rows=[line.split('=',1)[1].split() for line in (new/'stdout.log').read_text().splitlines() if line.startswith('captured_coarse_material_placement=')]
    if len(rows)!=1 or len(rows[0])!=12 or 'source placement capture complete' not in (new/'stderr.log').read_text():raise ValueError('No placement recovered')
    (out/'result.json').write_text(json.dumps(dict(placement_hex=rows[0],producer=digest(Path(json.loads((control/'launch.json').read_bytes())['argv'][0])),source=digest(original),control=digest(control/'launch.json'),capture_executable=digest(exe),commands=cmds,skin=digest(Path(cfg['argv'][cfg['argv'].index('--skin-payload')+1])),captured_mass_sha256=env['NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256'] if 'NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256' in env else env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_MASS_SHA256']),indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
