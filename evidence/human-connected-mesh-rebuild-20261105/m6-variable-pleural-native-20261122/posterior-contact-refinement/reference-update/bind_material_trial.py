"""Keep the same whole-body launch and replace only its authenticated Matter Metal library."""
import json
from pathlib import Path
import sys
from material_context import sha

def bind(parent,library,out):
    src=json.loads((parent/'launch.json').read_text());rec=json.loads((library/'identities.json').read_text())
    var='NUMI_MATTER_METALLIB_OVERRIDE';old=src['environment'][var];new=library/'NumiMatter.metallib'
    if not (src['pins'].pop(old)==sha(Path(old))==rec['control_metallib_sha256'] and sha(new)==rec['metallib_sha256'] and sha(library/'reference-update.patch')==rec['patch_sha256']):raise ValueError('Unmatched Metal owner')
    src['environment'][var]=str(new);src['pins'][str(new)]=sha(new)
    out.mkdir(exist_ok=False);(out/'launch.json').write_text(json.dumps(src,indent=2)+'\n')
if __name__=='__main__':bind(*(Path(s).resolve() for s in sys.argv[1:]))
