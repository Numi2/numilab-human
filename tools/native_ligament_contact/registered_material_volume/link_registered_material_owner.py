"""Link the selected Human scene to its resident Matter host/Metal pair."""
import json, subprocess, sys
from pathlib import Path
from emit_skin_source_registration import sha
src, release, out, app = map(lambda s: Path(s).resolve(), sys.argv[1:])
receipt = json.loads(release.read_text())
env = receipt['environment']
runtime = Path(env['DYLD_LIBRARY_PATH'])
lib = runtime / 'libmetalrobo.dylib'
shader = Path(env['NUMI_MATTER_METALLIB_OVERRIDE'])
for f in (lib, shader):
    if receipt['pins'].get(str(f)) != sha(f):
        raise ValueError('Unmatched release dependency ' + str(f))
host = runtime / 'app-runtime.mm.o'
original = Path(receipt['argv'][0])
if original.parent != runtime or not host.is_file():
    raise ValueError('Unrecognized paired host-object owner')
cmd = json.loads((src/'build-commands.json').read_text())[-1]
indices = [i for i,a in enumerate(cmd) if Path(a).name == 'runtime.mm.o']
if len(indices)!=1 or Path(cmd[cmd.index('-o')+1]) != src/'numi-human-native':
    raise ValueError('Unknown native app link layout')
out.mkdir(exist_ok=False)
for f in src.iterdir():
    if f.is_file() and f.suffix in ('.metal','.metalinc','.hpp','.metallib'):
        (out/f.name).hardlink_to(f)
(out/'metalrobo').symlink_to(src/'metalrobo',target_is_directory=True)
old = Path(cmd[indices[0]])
cmd[indices[0]] = str(host)
binary = out/'numi-human-native'
cmd[cmd.index('-o')+1] = str(binary)
subprocess.run(cmd,cwd=app/'build-native',check=True)
(out/'paired-link.json').write_text(json.dumps(dict(command=cmd,inputs={str(p):sha(p) for p in [original,old,host,shader,lib,src/'numi-human-native.o']},binary_sha256=sha(binary)),indent=2)+'\n')
