"""Compile both dispatch owners and the Metal kernel with the retained launch's flags."""
import argparse,concurrent.futures,json,subprocess
from pathlib import Path
from compile_connected_joint import digest
def once(s,old,new):
    if s.count(old)!=1:raise ValueError('Unmatched runtime binding '+old[:60])
    return s.replace(old,new)
def runtime(s,include):
    s=once(s,'releaseFloors=nil, releaseFailingRow=nil;','releaseFloors=nil, releaseFailingRow=nil, registeredBoundArguments=nil;')
    key='                        if(!candidate->releaseFloors || !candidate->releaseFailingRow)'
    s=once(s,key,'''                        candidate->registeredBoundArguments=privateScratch<std::uint32_t>(candidate->device,12u,valid,candidate->residentBytes);
                        if(!candidate->registeredBoundArguments || !candidate->releaseFloors || !candidate->releaseFailingRow)''')
    key='            "nm_fem_registered_release_reset",'
    s=once(s,key,key+''.join('\n            "nm_fem_registered_release_bound_'+k+'",' for k in ('begin','elements','finish')))
    for name in ['encodeHumanSupportLineSearchLimit','encodeVascularLineSearchLimit']:
        k='const auto '+name+' = [&] {'
        a=s.index(k);b=s.index('\n            };',a);frag=s[a:b]
        frag=once(frag,k,'const auto '+name+' = [&](id<MTLBuffer> conditional=nil) {')
        frag=once(frag,'                });','                },WorkDomain::generic,conditional,32u,nil,conditional?4u*sizeof(uint32_t):0u);')
        s=s[:a]+frag+s[b:]
    mark='            dispatchGroups32(\n                "nm_human_support_validate_final_line_search",'
    return once(s,mark,include+mark)

def run(base,metal,app,out):
    here=Path(__file__).resolve().parent
    out.mkdir(exist_ok=True);r=json.loads((base/'build.json').read_bytes());spec=json.loads((metal/'build.json').read_bytes());a=json.loads((app/'build.json').read_bytes())
    if digest(base/'runtime/libmetalrobo.dylib')!=a['library'] or digest(metal/'metal/NumiMatter.metallib')!=spec['shader']:raise ValueError('Unmatched runtime or shader')
    repl={}
    for folder,name in [('runtime','runtime.mm'),('app','app-runtime.mm')]:
        src=base/folder/name;dst=out/folder/name;dst.parent.mkdir(exist_ok=True)
        dst.write_text(runtime(src.read_text(),(here/'dispatch.inc').read_text()))
        for inc in src.parent.glob('*.inc'):
            target=dst.parent/inc.name
            if not target.exists():target.symlink_to(inc.resolve())
        for suffix in ('','.o'):repl[str(src)+suffix]=str(dst)+suffix
    repl[str(base/'runtime/libmetalrobo.dylib')]=str(out/'runtime/libmetalrobo.dylib')
    repl[str(app/'app/numi-human-native')]=str(out/'app/numi-human-native')
    src=metal/'metal/src';dest=out/'metal/src';(dest/'metal').mkdir(parents=True,exist_ok=True)
    for f in [*src.iterdir(),*(src/'metal').iterdir()]:
        if not f.is_file():continue
        target=dest/f.relative_to(src)
        if f.name=='fem.metalinc':
            code=f.read_text();marker='kernel void nm_fem_registered_release_reset'
            target.write_text(once(code,marker,(here/'bounded.metalinc').read_text()+'\n'+marker))
        elif not target.exists():target.symlink_to(f.resolve())
    commands=[[repl.get(v,v) for v in cmd] for cmd in [r['commands'][0],r['commands'][1],r['commands'][2],a['commands'][-1]]]
    commands.extend([[v.replace(str(metal/'metal')+'/',str(out/'metal')+'/') for v in cmd] for cmd in spec['commands']])
    def build(i):
        with (out/f'compile-{i}.log').open('w') as log:subprocess.run(commands[i],stdout=log,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(3) as ex:
        for job in [ex.submit(build,i) for i in (0,2,4)]:job.result()
    for i in (1,3,5):build(i)
    pins={str(p):digest(p) for p in [out/'runtime/libmetalrobo.dylib',out/'app/numi-human-native',out/'metal/NumiMatter.metallib']}
    (out/'build.json').write_text(json.dumps(dict(physical_parent=spec['shader'],app_parent=a['app'],library_parent=a['library'],pins=pins,commands=commands,runtime_source=digest(out/'runtime/runtime.mm'),shader_source=digest(dest/'metal/fem.metalinc')),indent=2)+'\n')
if __name__=='__main__':
    import sys
    run(*(Path(x).resolve() for x in sys.argv[1:]))
