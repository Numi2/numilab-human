"""Intersect the terminal residual witness with its independently captured material star."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from supported_material_admission import geom
from compile_connected_joint import digest

def run(attribution,volume,native,out):
    measurement=json.loads(attribution.read_bytes())
    v=measurement['spans'][0]
    if v['label']!='registered bonded skin and fat':raise ValueError('Unexpected largest residual owner')
    node=v['local']
    reference,loaded,end=[np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in (volume/'mesh.node',volume/'current.node',native/'physical-accepted.node')]
    elements=np.loadtxt(volume/'mesh.ele',skiprows=1,dtype=np.int32)
    indices=elements[:,1:5]-1
    ids=np.flatnonzero((indices==node).any(axis=1));t=indices[ids]
    if ids.size==0:raise ValueError('Empty witness region')
    cells=[]
    d0,s0=geom(reference,t);d1,s1=geom(loaded,t);d2,s2=geom(end,t)
    for row,e in enumerate(ids):
        cells.append(dict(zero_based_cell=int(e),region=int(elements[e,5]),vertices=(t[row]+1).tolist(),source_volume_m3=float(d0[row]),loaded_J=float(d1[row]/d0[row]),accepted_J=float(d2[row]/d0[row]),reference_quality=float(s0[row]),loaded_quality=float(s1[row]),accepted_quality=float(s2[row])))
    with (native/'native-traces/resting-body-motion-partition.csv').open() as f:
        name=next((r['body_name'] for r in csv.DictReader(f) if r['body_index']==v['peak_source']['donor_body']),None)
    bone=Path(json.loads((native/'launch.json').read_bytes())['argv'][3])
    manifest=bone.with_suffix('.manifest.json')
    b=json.loads(manifest.read_bytes())
    if digest(bone)!=b['payload']['sha256']:raise ValueError('Wrong bone identity')
    def registrations(item):
        if isinstance(item,dict):
            if item.get('core_body_index')==int(v['peak_source']['donor_body']) and 'myosim_body' in item:yield item
            for child in item.values():yield from registrations(child)
        if isinstance(item,list):
            for child in item:yield from registrations(child)
    records=list(registrations(b))
    names={r['myosim_body'] for r in records}
    if not names or len(names)!=1:raise ValueError('Ambiguous source registration')
    name=names.pop()
    nm=np.fromfile(volume/'node-mass.f64',dtype='<f8')
    if nm.shape!=(len(loaded),):raise ValueError('Wrong nodal mass layout')
    matter=dict(terminal_measurement=digest(attribution),source=v['peak_source'],donor_name=name,bone_manifest_sha256=digest(manifest),bone_source_members=sorted({r['member_id'] for r in records}),source_mass_kg=float(nm[node]),native_accepted_displacement_m=(end[node]-loaded[node]).tolist(),loaded_world_m=loaded[node].tolist(),cells=cells)
    out.write_text(json.dumps(matter,indent=2)+'\n')
    print(name,len(cells),'incident cells; minimum shape',min(s2),'J range',min(d2/d0),max(d2/d0),'mass',nm[node])
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('attribution','volume','native','out'):p.add_argument(n,type=Path)
    run(**vars(p.parse_args()))
