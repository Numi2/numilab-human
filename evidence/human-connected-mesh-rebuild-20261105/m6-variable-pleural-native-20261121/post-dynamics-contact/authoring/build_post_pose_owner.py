"""Rebuild the pinned owner's dylib with identical flags in an isolated directory."""
from pathlib import Path
import concurrent.futures,hashlib,json,shlex,subprocess,sys
from material_context import sha
app,out=map(Path,sys.argv[1:]);out=out.resolve();out.mkdir(exist_ok=True)
b=app/'build-native';src=app/'src/metal/MetalArticulatedOperator.mm'
old='''                hostMark("stand_phases");
                if (input.stand.numanXHumanMatterProgram.valid()) {'''
new='''                hostMark("stand_phases");
                // Stand has integrated q. Transaction-only post-dynamics consumers
                // also need poses and low coordinates at this new configuration.
                if (input.stand.numanXHumanMatterProgram.valid() ||
                    (pairedGeometry && input.stand.numanXTransactionProgram.valid())) {'''
text=src.read_text();assert text.count(old)==1
source=out/src.name;source.write_text(text.replace(old,new))
db={Path(c['output']):c for c in json.loads((b/'compile_commands.json').read_text())}
link=shlex.split(subprocess.check_output(['ninja','-t','commands','lib/libmetalrobo.dylib'],cwd=b,text=True).splitlines()[-1]);assert link[:2]==[':','&&'] and link[-2:]==['&&',':'];link=link[2:-2]
repeat=list(link);objects=[];spec=[]
for i,x in enumerate(link):
    if not x.endswith('.o'):continue
    rec=db[(b/x).resolve()];original=Path(rec['file']);op=out/(str(i)+'-'+original.name+'.o')
    cmd=shlex.split(rec['command']);cmd[cmd.index('-o')+1]=str(op)
    if original==src:
        cmd[cmd.index('-c')+1]=str(source);cmd[1:1]=['-iquote',str(src.parent)]
        orig=list(cmd);orig[orig.index('-c')+1]=str(src);sibling=out/'control-operator.o';orig[orig.index('-o')+1]=str(sibling);spec.append((sibling,orig));repeat[i]=str(sibling)
    else:repeat[i]=str(op)
    link[i]=str(op);spec.append((op,cmd));objects.append((str(original),sha(original)))
def compile_one(item):
    op,cmd=item
    if op.is_file():return
    with op.with_suffix('.log').open('w') as log:subprocess.run(cmd,cwd=b,stdout=log,stderr=log,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile_one,spec))
for cmd,variant in [(repeat,'control'),(link,'refreshed')]:
    path=out/variant/'libmetalrobo.dylib';path.parent.mkdir(exist_ok=True);cmd[cmd.index('-o')+1]=str(path);subprocess.run(cmd,cwd=b,check=True)
(out/'endpoint-owner.json').write_text(json.dumps(dict(pinned_library=sha(b/'lib/libmetalrobo.dylib'),control=sha(out/'control/libmetalrobo.dylib'),refreshed=sha(out/'refreshed/libmetalrobo.dylib'),operator_before=sha(src),operator_after=sha(source),sources=objects,compile_commands=sha(b/'compile_commands.json')),indent=2)+'\n')
