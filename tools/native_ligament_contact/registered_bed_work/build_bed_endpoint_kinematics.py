"""Rebind the existing articulated FK pass before the endpoint contact check."""
from pathlib import Path
import json,shlex,subprocess,sys
from material_context import sha
from export_contact_columns import change
base,owner,build=map(lambda x:Path(x).resolve(),sys.argv[1:])
read=lambda p:json.loads(p.read_text())
ids=read(base/'identities.json')
if any(sha(base/k)!=v for k,v in ids.items()):raise ValueError('changed endpoint source')
subprocess.run(['cp','-cR',base,owner],check=True)
config=next(x for x in read(build/'compile_commands.json') if x['file'].endswith('/MetalArticulatedOperator.mm'))
src=Path(config['file']);new=owner/src.name; original=src.read_text()
anchor='            const char* geometryOverlapSetting ='
newsrc=change(original,anchor,'''            const bool requireEndpointBedKinematics =
                input.stand.numanXTransactionProgram.valid() &&
                input.stand.supportGeometryProgram.valid() &&
                std::getenv("NUMI_HUMAN_RESOLVED_BED_WITNESS") != nullptr;
'''+anchor)
old='hostMark("stand_phases");\n                if (input.stand.numanXHumanMatterProgram.valid()) {'
newsrc=change(newsrc,old,old.replace('.valid())','.valid() || requireEndpointBedKinematics)'))
new.write_text(newsrc)
obj=owner/'MetalArticulatedOperator.mm.o'
compile=shlex.split(config['command'])
compile[compile.index('-o')+1]=str(obj)
compile[compile.index('-c')+1]=str(new)
compile[1:1]=['-iquote',str(src.parent)]
link=[x.replace(str(base)+'/',str(owner)+'/') for x in read(base/'build-commands.json')[-1]]
link.insert(link.index('lib/libmetalrobo.dylib'),str(obj))
for cmd in [compile,link]:subprocess.run(cmd,cwd=build,check=True)
(owner/'endpoint-articulated-build.json').write_text(json.dumps(dict(source=sha(src),commands=[compile,link],borrowed_library=sha(build/'lib/libmetalrobo.dylib')),indent=2)+'\n')
(owner/'identities.json').write_text(json.dumps({k:sha(owner/k) for k in sorted(set(ids)|{src.name,'endpoint-articulated-build.json'})},indent=2)+'\n')
