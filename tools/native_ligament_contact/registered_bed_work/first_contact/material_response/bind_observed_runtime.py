"""Bind the sparse diagnostic to the shared physical runtime as well as the scene startup owner."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import difflib,json,os,shutil,subprocess,sys
from compile_connected_joint import digest

def run(prev,out):
    source=prev.parent/'incident-direction';original=prev.parent/'bound-incident-app';out.mkdir()
    ap=out/'app';rt=out/'runtime';ap.mkdir();rt.mkdir()
    (out/'metal').symlink_to(prev/'metal',target_is_directory=True)
    os.link(source/'runtime/bed_response.inc',rt/'bed_response.inc')
    ref=(original/'app-runtime.mm').read_text();observed=(prev/'app/app-runtime.mm').read_text()
    # Both owners contain the same selected trace, with different ABI entry points.
    begin='const NSUInteger partialBytes = NSUInteger(trials)*maximumTiles*sizeof(nm_float4);'
    finish='std::ostringstream line;\n                        line<<"fem_backtracking_trials='
    def region(s):
        i=s.index(begin);e=s.index(finish,i);return s[i:e]
    a=region(ref);b=region(observed).replace('if(traceVolume){\n                            const uint32_t* sparse=', '''if(!traceVolume)std::fprintf(stderr,"registered_volume_direction_outcomes root=%u iteration=%u unavailable=1\\n",root,iteration);
                        if(traceVolume){
                            const uint32_t* sparse=''')
    for fn,where,control in [('runtime.mm',rt,source/'runtime/runtime.mm'),('app-runtime.mm',ap,original/'app-runtime.mm')]:
        text=control.read_text()
        if text.count(a)!=1:raise ValueError('Ambiguous physical trace region')
        (where/fn).write_text(text.replace(a,b))
    shared=json.loads((source/'build.json').read_bytes());app=json.loads((prev/'app/build.json').read_bytes())
    def remap(cmd):
        return [w.replace(str(source/'runtime')+'/',str(rt)+'/').replace(str(prev/'runtime')+'/',str(rt)+'/').replace(str(prev/'app')+'/',str(ap)+'/')for w in cmd]
    cpu=[remap(c) for c in shared['commands'][:2]];exe=[remap(c)for c in app['commands']]
    def invoke(name,c):
        with (out/(name+'.log')).open('w')as f:subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(2) as pool:
        a=pool.submit(invoke,'runtime-compile',cpu[0]);b=pool.submit(invoke,'app-compile',exe[0]);a.result();b.result()
    invoke('runtime-link',cpu[1]);invoke('app-link',exe[1])
    meta=json.loads((prev/'build.json').read_bytes());meta.update(runtime=digest(rt/'libmetalrobo.dylib'),host_source=digest(rt/'runtime.mm'),commands=cpu+exe)
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
    app.update(runtime=meta['runtime'],commands=exe,source=digest(ap/'app-runtime.mm'),app=digest(ap/'numi-human-native'))
    (ap/'build.json').write_text(json.dumps(app,indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
