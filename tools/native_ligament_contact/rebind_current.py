from pathlib import Path
import hashlib,json,subprocess,sys
from numilab_human.respiratory_bone_rebind import rebind, write_bundle, file_hash

def run(current, parent_scene, output):
    doc=json.loads(current.read_text())
    parent=json.loads(parent_scene.read_text())
    args=doc['argv'];idx=args.index('--resting-anatomy-receipt')+1
    original=Path(args[idx])
    updated,proof=rebind(Path(parent['argv'][3]),Path(args[3]),Path(args[args.index('--torso-anatomy-payload')+1]),original)
    bundle=output.parent/'respiratory-binding'
    bundle.mkdir(exist_ok=True)
    receipt=bundle/'resting-anatomy-receipt.json'
    write_bundle(updated,proof,original,receipt,bundle/'verified.json')
    args[idx]=str(receipt)
    doc['hashes'][str(receipt)]=file_hash(receipt)
    output.write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
