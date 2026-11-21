"""Compile the full owner's production material stamp from its actual lowering."""
from pathlib import Path
import sys, json, subprocess, shutil, hashlib, re
from concurrent.futures import ThreadPoolExecutor
from emit_skin_source_registration import sha
base,whole,selected,app,out,launch=map(lambda s:Path(s).resolve(),sys.argv[1:])
original=base/'guarded-runtime-admission-build'; prior=base/'pressure-with-abdominal-cavity-build'
native=app/'matter/src'; emitted=selected/'generated_stateless_tangent.metalinc'
stock=native/'metal'/emitted.name
shader_source=base/'contact-prefix-certification/src/metal'/emitted.name
a,b=stock.read_bytes(),emitted.read_bytes()
if a!=shader_source.read_bytes():raise ValueError('Authoritative Metal source differs from the owner')
def parse(data):
    head,sep,body=data.partition(b'#else\n')
    m=re.fullmatch(rb'// Generated exact scalar-operation lowering\. Bound to compiled material/program bytes\.\n#ifndef __METAL_VERSION__\nstatic constexpr std::uint64_t nmNativeTangentSourceHash=(\d+)ull;\n',head)
    if not m or not sep or not body:raise ValueError('Unknown generated lowering')
    return m[1].decode(),body
old,metal=parse(a);new,current=parse(b)
if metal!=current:raise ValueError('Metal operations changed; compile the new shader bundle')
j=json.loads(launch.read_text());env=j['environment'];shader=Path(env['NUMI_MATTER_METALLIB_OVERRIDE']);lib=original/'libmetalrobo.dylib'
if sha(shader)!=j['pins'][str(shader)] or sha(lib)!=j['pins'][str(lib)]: raise ValueError('Release identity changed')
out.mkdir(exist_ok=False);(out/'metal').mkdir()
(out/'metal'/emitted.name).hardlink_to(emitted)
(out/'native_tangent_source.hpp').hardlink_to(native/'native_tangent_source.hpp')
shared=json.loads((prior/'build-commands.json').read_text())
static_ref=base.parent/'resting-clearance-and-atrium-007/shared-fascia-pressure-build'
static=json.loads((static_ref/'build-commands.json').read_text())[0]
build=app/'build-native'
jobs=[]
for name,cmd,root in [('runtime.mm',shared[0],prior),('app-runtime.mm',static,static_ref)]:
    (out/name).hardlink_to(original/name)
    jobs.append([str(out/(name+'.o')) if x==str(root/'runtime.mm.o') else str(out/name) if x==str(root/'runtime.mm') else x for x in cmd])
    if str(out/name) not in jobs[-1] or str(out/(name+'.o')) not in jobs[-1]:raise ValueError('Unrecognized host compiler command')
def build_one(cmd):subprocess.run(cmd,cwd=build,check=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(build_one,jobs))
link=[str(out/Path(x).name) if Path(x).parent==prior and Path(x).name in ('runtime.mm.o','libmetalrobo.dylib') else x for x in shared[-1]]
if str(out/'libmetalrobo.dylib') not in link or str(out/'runtime.mm.o') not in link:raise ValueError('Unknown shared link')
build_one(link)
c=json.loads((whole/'paired-link.json').read_text())['command']
for i,x in enumerate(c):
    if Path(x)==original/'app-runtime.mm.o':c[i]=str(out/'app-runtime.mm.o')
c[c.index('-o')+1]=str(out/'numi-human-native')
assert str(out/'app-runtime.mm.o') in c
build_one(c)
for p in whole.iterdir():
    if p.name=='metalrobo':(out/p.name).symlink_to(p.resolve(),target_is_directory=True)
    elif p.suffix in ('.metal','.metalinc','.hpp','.metallib') and not (out/p.name).exists():(out/p.name).hardlink_to(p)
j['environment']['DYLD_LIBRARY_PATH']=str(out)
j['pins'][str(out/'libmetalrobo.dylib')]=sha(out/'libmetalrobo.dylib')
j['pins'][str(emitted)]=sha(emitted)
(out/'launch.json').write_text(json.dumps(j,indent=2)+'\n')
(out/'material-lowering-proof.json').write_text(json.dumps(dict(prior_world_hash=old,selected_world_hash=new,metal_operations_byte_identical=True,metal_operations_sha256=hashlib.sha256(current).hexdigest(),original_metal_image_sha256=sha(shader),sources={str(p):sha(p) for p in [emitted,stock,shader_source,native/'native_tangent_source.hpp',original/'app-runtime.mm',original/'runtime.mm']},commands=[*jobs,link,c],binary_sha256=sha(out/'numi-human-native'),dynamic_library_sha256=sha(out/'libmetalrobo.dylib')),indent=2)+'\n')
