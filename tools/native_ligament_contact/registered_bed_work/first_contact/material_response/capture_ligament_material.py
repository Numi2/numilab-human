"""Read the admitted control's actual native ligament triangle, endpoint, and elastic rest length."""
import json,os,subprocess,sys
from pathlib import Path
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(author,control,out):
    app='numilab_human_myosim_visual_probe.mm';src=(author/app).read_text();marker='registeredLigaments.reserve(catalog.size());'
    if src.count(marker)!=1:raise ValueError('Unknown native catalog owner')
    src=src.replace(marker,marker+'''
        std::ofstream materialRows(std::getenv("NUMI_HUMAN_LIGAMENT_MATERIAL_CAPTURE"));
        require(materialRows.good(),"ligament material source capture path is invalid");materialRows<<std::setprecision(17);
    ''')
    k='spec,model,bodies,surfaces,sourceBodyMap,ownership));'
    if src.count(k)!=1:raise ValueError('Unknown native source cook')
    src=src.replace(k,k+'''
                const auto& b=registeredLigaments.back();const auto& p=b.provenance;
                materialRows<<p.name<<" "<<p.side<<" "<<p.body0<<" "<<p.stable0<<" "<<p.triangle0<<" "<<p.body1<<" "<<p.stable1<<" "<<p.triangle1<<" "<<p.referenceLengthM;
                for(const auto& v : {b.row.firstLocalPoint,b.row.secondLocalPoint})materialRows<<" "<<v.x<<" "<<v.y<<" "<<v.z;
                materialRows<<" "<<p.stiffnessNPerM<<" "<<p.areaM2<<" "<<p.residual0M<<" "<<p.residual1M<<"\\n";
                if(registeredLigaments.size()==catalog.size()){
                    materialRows.flush();require(materialRows.good(),"native catalog material capture failed");
                    std::cout<<"source_ligament_material_capture_complete="<<registeredLigaments.size()<<std::endl;
                }
    ''')
    out.mkdir();(out/app).write_text(src);cmds=json.loads((author/'build.json').read_bytes())['commands']
    for f in author.iterdir():
        if f.name!=app and f.suffix in ('.h','.mm','.hpp','.inc'):os.link(f,out/f.name)
    exe=out/'numi-human-native'
    for i,cmd in enumerate(cmds):
        cmds[i]=[str(out/Path(v).name) if Path(v).parent==author and Path(v).name in (app,'numi-human-native','numi-human-native.o') else v for v in cmd]
        with (out/'compiler.log').open('a') as log:subprocess.run(cmds[i],stdout=log,stderr=subprocess.STDOUT,check=True)
    cfg=json.loads((control/'launch.json').read_bytes());argv=cfg['argv'];old=argv[0]
    if Path(old)!=author/'numi-human-native' or digest(Path(old))!=json.loads((author/'build.json').read_bytes())['executable_sha256']:raise ValueError('Unmatched ligament native control')
    argv[0]=str(exe);argv[4]=str(out/'native-traces');env={k:v.replace(str(control)+'/',str(out)+'/') for k,v in cfg['environment'].items()}
    env['NUMI_HUMAN_LIGAMENT_MATERIAL_CAPTURE']=str(out/'native-attachments.txt')
    # Stop after extraction, before expensive physics: the capture has no motion claim.
    source=(out/app).read_text()
    cfg['source_capture']=dict(original=digest(author/app),application=digest(out/app),binary=digest(exe),commands=cmds)
    cfg['environment']=env;cfg['pins'].pop(old,None);cfg['pins'][str(exe)]=digest(exe);(out/'launch.json').write_text(json.dumps(cfg,indent=2)+'\n')
    # The renderer enforces these checks after support preparation. The catalog becomes available immediately before integration.
    import signal,time
    (out/'native-traces').mkdir(exist_ok=True)
    with (out/'stdout.log').open('w') as so,(out/'stderr.log').open('w') as se:
        with subprocess.Popen(argv,cwd=cfg['cwd'],env={k:v for k,v in os.environ.items() if not k.startswith(('NUMI_','NM_'))}|env,stdout=so,stderr=se) as child:
            end=time.monotonic()+100;ok=False
            while child.poll() is None and time.monotonic()<end:
                if 'source_ligament_material_capture_complete=' in (out/'stdout.log').read_text():
                    ok=True;break
                time.sleep(.25)
            if child.poll() is None:child.terminate()
            child.wait(timeout=12)
    if not ok:raise ValueError('Native source ligament capture did not finish')
    print('Captured',len((out/'native-attachments.txt').read_text().splitlines()),'ligaments')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
