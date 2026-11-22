"""Identify loaded posterior nodes by registered area and initial indentation on the unchanged plane."""
import csv,json,sys
from pathlib import Path
import numpy as np
from evaluate_registered_bed_transfer import digest
def patch(full,placement):
    j=json.loads(full.read_text());p=json.loads(placement.read_text());e=j['environment'];key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def file(name,hashkey):
        src=Path(e[key+name]);assert digest(src)==e[key+hashkey];return src
    xyz=np.loadtxt(file('TETGEN_NODE','TETGEN_NODE_SHA256'),skiprows=1)[:,1:]
    faces=np.loadtxt(file('SURFACE_O_FACES_CSV','SURFACE_O_FACES_SHA256'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype=int)
    ids=np.full(faces.max()+1,-1,dtype=int)
    with file('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256').open() as f:
        for v in csv.DictReader(f):
            if v['registered_surface_layer']=='O':
                i=int(v['registered_surface_vertex_row']);assert ids[i]==-1;ids[i]=int(v['joint_node_id'])-1
    assert np.all(ids>=0);tri=xyz[ids[faces]]
    area=.5*np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    vectors=np.zeros_like(xyz[ids]);nodal=np.zeros(len(ids))
    for k in range(3):
        np.add.at(vectors,faces[:,k],area/3);np.add.at(nodal,faces[:,k],np.linalg.norm(area,axis=1)/3)
    depth=-float(p['construction_translation_m'][2]);penetration=depth-xyz[ids,2]
    downward=-vectors[:,2]/np.maximum(nodal,1e-30)
    active=(penetration>0)&(downward>.55);n=ids[active];order=n.argsort()
    weights=(-vectors[active,2])*penetration[active]
    return n[order],weights[order],dict(descent_m=depth,posterior_normal_fraction=.55,eligible_nodes=len(n),
        represented_projected_nodal_area_cm2=float(-vectors[active,2].sum()*1e4),reference_outer_sha256=e[key+'SURFACE_O_NODES_SHA256'],
        placement_sha256=digest(placement))
