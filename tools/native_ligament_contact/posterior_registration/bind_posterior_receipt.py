"""Rebind an edited NHTISS4 carrier only after proving every indexed parent row is byte-identical."""
import json,struct,sys
from pathlib import Path
import numpy as np
from material_context import sha
from stage_bound_assets import stage
from guard_material_rebind import validate
def run(folder):
    config=folder/'config.json';cfg=json.loads(config.read_text());args=cfg['argv']
    old=Path(json.loads((folder/'source.json').read_text())['parent']);new=Path(args[args.index('--soft-tissue-payload')+1])
    slot=args.index('--resting-anatomy-receipt')+1
    parent=Path(args[slot]);receipt=json.loads(parent.read_text());b=receipt['passive_tissue_parent_bindings']
    if b['child_payload_sha256']!=sha(old):raise ValueError('Wrong parent carrier SHA')
    before=old.read_bytes();after=new.read_bytes()
    _,abi,nr,nb,nv,ni,_,_=struct.unpack_from('<8s6I32s',before);at=64+32*nr+36*nb
    allowed={29,30};bound={r['stable_id'] for r in b['surfaces']}
    if bound & allowed:raise ValueError('Changed separate refined ancestry')
    rows=np.ndarray((nr,8),'<u4',before,64)
    validate(before,after,nr,nb,nv,rows,json.loads((folder/'source.json').read_text()))
    b['child_payload_sha256']=sha(new)
    active=receipt['provenance']['native_muscle_surfaces']
    if active['sha256']!=sha(old):raise ValueError('Source provenance mismatch')
    if Path(active['payload_path']).resolve()!=old.resolve():raise ValueError('Provenance points to different source')
    active['sha256']=sha(new);active['payload_path']=str(new)
    manifest=new.with_suffix('.manifest.json')
    active['manifest_path']=str(manifest);active['manifest_sha256']=sha(manifest)
    receipt['provenance']['bilateral_gluteal_registration']=dict(parent_payload_sha256=sha(old),candidate_payload_sha256=sha(new),parent_receipt_sha256=sha(parent),edited_source_ids=sorted(allowed),unchanged_parent_bound_surfaces=sorted(bound))
    dest=folder/'resting-anatomy-receipt.json';dest.write_text(json.dumps(receipt,separators=(',',':'))+'\n')
    for name,digest in stage(parent,dest).items():cfg['hashes'][str((dest.parent/name).resolve())]=digest
    args[slot]=str(dest.resolve());cfg['hashes'][str(dest.resolve())]=sha(dest)
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    (folder/'lineage.json').write_text(json.dumps(receipt['provenance']['bilateral_gluteal_registration']|{'receipt_sha256':sha(dest)},indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]))
