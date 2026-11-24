"""Resolve and verify relative, hash-pinned dependencies when relocating the anatomy receipt."""
from pathlib import Path
import json
from compile_connected_joint import digest

def preserve_receipt_assets(original,rebound):
    doc=json.loads(rebound.read_bytes())
    expected={}
    def visit(v):
        if isinstance(v,dict):
            p=v.get('path');h=v.get('sha256')
            if isinstance(p,str) and isinstance(h,str)and not Path(p).is_absolute():
                if '..' in Path(p).parts or (p in expected and expected[p]!=h):raise ValueError('Conflicting receipt dependency')
                expected[p]=h
            for item in v.values():
                if isinstance(item,(dict,list)):visit(item)
        elif isinstance(v,list):
            for item in v:
                if isinstance(item,(dict,list)):visit(item)
    visit(doc['functional_bindings'])
    visit(doc['provenance']['cardiac_geometry_binding']['common_field'])
    if not expected:raise ValueError('Missing native common-field files')
    for name,h in expected.items():
        src=original.parent/name;dst=rebound.parent/name
        if digest(src)!=h:raise ValueError('Changed receipt dependency '+name)
        if not dst.exists():dst.parent.mkdir(exist_ok=True,parents=True);dst.symlink_to(src.resolve())
        if digest(dst)!=h:raise ValueError('Invalid relocated receipt dependency '+name)
    return expected
