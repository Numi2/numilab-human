"""Resolve all three binary assets through the active native common-field binding."""
import json,os,sys
from pathlib import Path
from material_context import sha
def stage(parent,receipt):
    field=json.loads(receipt.read_text())['provenance']['cardiac_geometry_binding']['common_field']
    layout={'map':field['map']['record_count']*field['map']['record_stride_bytes'],
            'polynomials':4*field['polynomials']['term_count']*field['polynomials']['volume_count'],
            'domain_boxes':field['domain_boxes']['count']*field['domain_boxes']['record_stride_bytes']}
    result={}
    for key,size in layout.items():
        desc=field[key];name=desc['path']
        if Path(name).name!=name:raise ValueError('Nonlocal field asset '+name)
        src=parent.parent/name;dst=receipt.parent/name;digest=desc['sha256']
        if src.stat().st_size!=size or sha(src)!=digest:raise ValueError('Unbound active '+key)
        if not dst.exists():os.link(src,dst)
        if sha(dst)!=digest:raise ValueError('Staging mismatch')
        result[name]=digest
    return result
if __name__=='__main__':print(json.dumps(stage(*map(Path,sys.argv[1:]))))
