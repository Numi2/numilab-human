"""Build all compiler, package and runtime consumers of the same admission certificate."""
import hashlib,json,shlex,shutil,subprocess,sys
from pathlib import Path
from add_exact_material_edges import write
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
app,prior,owner,witness,config,out=map(lambda a:Path(a).resolve(),sys.argv[1:])
out.mkdir(exist_ok=False)
src=app/'matter/src'; build=app/'build-native'
write(src/'fem_reference_geometry.hpp',out/'fem_reference_geometry.hpp')
db=json.loads((build/'compile_commands.json').read_text())
proof=json.loads((prior/'material-lowering-proof.json').read_text())['commands']
commands=[]
def run(c):
    commands.append(c);subprocess.run(c,cwd=build,check=True)
for f in ('native_tangent_source.hpp','metal'):
    (out/f).symlink_to(prior/f)
for name in ('compiler_impl.cpp','validation_impl.cpp'):
    path=src/name
    temp=out/name
    shutil.copyfile(witness/name if name=='compiler_impl.cpp' else path,temp)
    e=next(q for q in db if q['file']==str(path))
    c=shlex.split(e['command']);c[c.index('-c')+1]=str(temp);c[c.index('-o')+1]=str(out/(name+'.o'))
    c[1:1]=['-iquote',str(src)];run(c)
for source in proof[:2]:
    c=list(source);f=Path(c[c.index('-c')+1]);temp=out/f.name
    shutil.copyfile(f,temp);c[c.index('-c')+1]=str(temp);c[c.index('-o')+1]=str(out/(f.name+'.o'));run(c)
lib=list(proof[2])
for index,v in enumerate(lib):
    if Path(v).name in ('validation_impl.cpp.o','compiler_impl.cpp.o','runtime.mm.o','libmetalrobo.dylib'):
        lib[index]=str(out/Path(v).name)
run(lib)
c=json.loads((owner/'identities.json').read_text())['commands'][1];c[c.index('-o')+1]=str(out/'numi-human-native')
c[c.index(str(prior/'app-runtime.mm.o'))]=str(out/'app-runtime.mm.o')
c[c.index('lib/libmetalrobo.dylib')]=str(out/'libmetalrobo.dylib');run(c)
launch=json.loads(config.read_text())
launch['environment']['DYLD_LIBRARY_PATH']=str(out);launch['pins'][str(out/'libmetalrobo.dylib')]=sha(out/'libmetalrobo.dylib')
launch['pins'][str(out/'numi-human-native')]=sha(out/'numi-human-native')
(out/'config.json').write_text(json.dumps(launch,indent=2)+'\n')
files=['fem_reference_geometry.hpp','compiler_impl.cpp','validation_impl.cpp','runtime.mm','app-runtime.mm','numi-human-native','libmetalrobo.dylib']
(out/'identities.json').write_text(json.dumps({'commands':commands,'sha256':{n:sha(out/n) for n in files},'material_shader_unchanged':True},indent=2)+'\n')
