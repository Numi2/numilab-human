"""Use the independently captured source pose when cooking the immutable historical mass parent."""
import json,os,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native
def run(base,control,capture,out):
    data=json.loads((capture/'result.json').read_bytes());pose=data['placement_hex'];cfg=json.loads((control/'launch.json').read_bytes());env=cfg['environment']
    if data['captured_mass_sha256']!=env['NUMI_H706_FRACTIONAL_SOURCE330_MASS_SHA256'] or data['skin']==digest(Path(cfg['argv'][cfg['argv'].index('--skin-payload')+1])):raise ValueError('Expected a separately bound historical mass source')
    src=base/'app';out.mkdir();app=out/'app';app.mkdir();unit='numilab_human_myosim_visual_probe.mm'
    for f in src.iterdir():
        if f.name!=unit and f.suffix in ('.hpp','.mm','.h','.inc'):os.link(f.resolve(),app/f.name)
    s=(src/unit).read_text();key='            for(auto& p:region.worldPoints)p=skinPlacement.point(p);'
    if s.count(key)!=1:raise ValueError('Cannot locate original layered source placement')
    insert='''            if(selectedParent && registeredVolumeInstallRequested && registeredFractionalMassSupportRequested){
                // These coarse rows solely reproduce the historical physical mass ledger.
                // The registered child has independent rest, posed coordinates and current donor frames.
                const char* sourcePacket=std::getenv("NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_PARENT_PACKET_CSV");
                require(sourcePacket && loadedKneeSHA256Hex(loadedKneeFileSHA256(sourcePacket))=="MASS_PIN",
                        "historical parent material source changed");
                const std::array<double,9> registeredRotation{ROTATION};
                const std::array<double,3> registeredTranslation{TRANSLATION};
                auto originalPlacement=skinPlacement;
                originalPlacement.rotation=registeredRotation;originalPlacement.translation=registeredTranslation;
                double maxDistance=0.;
                for(const auto& p:region.worldPoints)
                    maxDistance=std::max(maxDistance,norm(sub(originalPlacement.point(p),skinPlacement.point(p))));
                require(maxDistance<=2e-5,"historical parent source placement is outside the registered surface envelope");
                skinPlacement=originalPlacement;
                std::cout<<std::setprecision(17)<<"historical_parent_placement=verified source_sha256=SOURCE_PIN"
                    <<" maximum_registered_position_difference_m="<<maxDistance<<std::endl;
            }
'''.replace('ROTATION',','.join(pose[:9])).replace('TRANSLATION',','.join(pose[9:])).replace('MASS_PIN',data['captured_mass_sha256']).replace('SOURCE_PIN',data['source'])
    s=s.replace(key,insert+key);(app/unit).write_text(s)
    cmds=[[v.replace(str(src)+'/',str(app)+'/') for v in c] for c in json.loads((base/'build.json').read_bytes())['commands']]
    with (out/'build.log').open('w') as f:
        for c in cmds:subprocess.run(c,check=True,stdout=f,stderr=subprocess.STDOUT)
    exe=app/'numi-human-native';(out/'build.json').write_text(json.dumps(dict(commands=cmds,source=digest(src/unit),current=digest(app/unit),parent_placement=digest(capture/'result.json'),executable=digest(exe)),indent=2)+'\n')
    job=out/'scene';job.mkdir();env={k:v.replace(str(control)+'/',str(job)+'/') for k,v in env.items()};cfg['argv'][0]=str(exe);cfg['argv'][4]=str(job/'native-traces');cfg['environment']=env;cfg['parent_launch']=digest(control/'launch.json');cfg['pins'][str(exe)]=digest(exe)
    (job/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(execute_native(cfg['argv'],cfg['cwd'],{k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_'))}|env,job,minimum_free_bytes=4*2**30,timeout_s=400))
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
