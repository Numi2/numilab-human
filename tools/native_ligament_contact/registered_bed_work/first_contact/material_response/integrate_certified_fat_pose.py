"""Rebind the certified material starting pose into the full registered scene."""
import sys,json,os
from pathlib import Path
from compile_connected_joint import digest
from register_loaded_cortical_interfaces import run as project
from compile_cortical_skin_contact import run as compile_contact
from build_repaired_native_pose import run as build
from run_repaired_fat_scene import run as execute
def run(job,h,lib,qual,out):
    rd=lambda p:json.loads(p.read_bytes());out.mkdir();mat=out/'material';mat.mkdir()
    native=job/'deep-fat-native';q=rd(qual/'result.json');original=rd(native/'result.json')
    if q['reference']!=original['material'] or q['original']!=original['initial'] or q['updated']!=digest(qual/'current.node') or any(any(r) for r in q['exact_layers']):raise ValueError('Unqualified physical pose')
    for k in ('mesh.node','mesh.ele','cell-mass.f64','node-mass.f64'):os.link(native/k,mat/k)
    os.link(qual/'current.node',mat/'current.node');original.update(initial=q['updated'],loaded_jacobian_range=q['loaded_jacobian_range'],initial_interval_certificate=digest(qual/'result.json'));original['layer_intersections']['starting']=q['exact_layers']
    (mat/'result.json').write_text(json.dumps(original,indent=2)+'\n')
    project(h,job,lib,out/'cortex',mat/'current.node');compile_contact(h,job,job/'physical-weight-frames',out/'contact',mat,mat/'current.node',out/'cortex')
    build(job/'registered-organ-viewer',mat/'current.node',out/'native',out/'contact',job/'registered-organ-scene')
    execute(job,mat,job/'deep-fat-ledger',job/'deep-fat-p1',job/'deep-fat-registered-tables',out/'native',out/'scene')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
