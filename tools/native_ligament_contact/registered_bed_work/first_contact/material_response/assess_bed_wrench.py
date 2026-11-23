"""Audit actual bed wrench and the normal-force support polygon of the registered release."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from register_material_surfaces import sha


def supportable(xy, center):
    if len(xy)<3 or np.linalg.matrix_rank(xy-xy[0])<2:
        if len(xy)==0:return False
        delta=xy-xy[0]
        farthest=delta[np.argmax(np.linalg.norm(delta,axis=1))]
        length=np.linalg.norm(farthest)
        if not length:return np.linalg.norm(xy[0]-center)<1e-10
        direction=farthest/length
        t=(center-xy[0])@direction
        projected=xy@direction-xy[0]@direction
        return bool(np.linalg.norm(center-xy[0]-t*direction)<1e-10 and projected.min()<=t<=projected.max())
    h=ConvexHull(xy)
    return bool(np.max(h.equations[:,:2]@center+h.equations[:,2])<=1e-12)


def assess(run,out):
    stdout=(run/'stdout.log').read_text().splitlines()
    def match(prefix):
        rows=[l for l in stdout if l.startswith(prefix)]
        if len(rows)!=1:raise ValueError('Missing or repeated '+prefix)
        return rows[0]
    def field(line,name):
        return re.search(r'(?:^| )'+name+r'=(\S+)',line)[1]
    masses=[json.loads(l.split('=',1)[1]) for l in stdout if l.startswith('initial_gravity_component={')]
    if [r['object'] for r in masses]!=list(range(-1,22)):raise ValueError('Incomplete body mass')
    mass=sum(r['mass_kg'] for r in masses)
    center=np.sum([r['first_moment_kg_m'] for r in masses],axis=0)/mass
    bed=match('resting_fem_skin_bed_source=')
    if field(bed,'normal')!='0,0,1':raise ValueError('Expected a horizontal bed')
    plane=float(field(bed,'plane_offset_m'))
    env=json.loads((run/'launch.json').read_bytes())['environment']
    key='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_'
    mapping=Path(env[key+'CSV'])
    if sha(mapping)!=env[key+'SHA256']:raise ValueError('Unregistered surface')
    with mapping.open() as f:
        outer=np.array([int(r['joint_node_id'])-1 for r in csv.DictReader(f) if r['registered_surface_layer']=='O'])
    raw=np.loadtxt(run/'physical-reference.node',skiprows=1)
    if not np.array_equal(raw[:,0],np.arange(1,len(raw)+1)) or len(np.unique(outer))!=int(field(bed,'outer_nodes')):
        raise ValueError('Incomplete surface')
    P=raw[outer,1:4]
    order=np.argsort(P[:,2],kind='stable')
    sorted_points=P[order]
    lo,hi=0,len(P)
    if not supportable(sorted_points[:,:2],center[:2]):raise ValueError('Weight line outside entire footprint')
    while lo+1<hi:
        mid=(lo+hi)//2
        if supportable(sorted_points[:mid,:2],center[:2]):hi=mid
        else:lo=mid
    impulse=match('resting_fem_skin_bed_accepted_root_reaction ')
    I=np.array(json.loads(field(impulse,'impulse_on_plane_xyz_Ns')))
    M=np.array(json.loads(field(impulse,'angular_impulse_on_plane_xyz_Nms')))
    # Moments about the world origin on a plane at z=plane. Includes tangential impulse.
    if I[2]>=0:raise ValueError('No compressive normal reaction')
    cop=np.array([(plane*I[0]-M[1])/I[2],(M[0]+plane*I[1])/I[2]])
    force=float(field(impulse,'mean_normal_force_N'))
    weight=mass*9.81
    gap=sorted_points[hi-1,2]-plane
    record=dict(physical_start_sha256=sha(run/'physical-reference.node'),mass_kg=mass,center_of_mass_m=center.tolist(),
       plane_height_m=plane,bed_reaction_N=force,weight_N=weight,bed_force_fraction=force/weight,
       angular_impulse_on_plane_Nms=M.tolist(),linear_impulse_on_plane_Ns=I.tolist(),bed_center_of_pressure_xy_m=cop.tolist(),
       planar_distance_from_weight_line_m=float(np.linalg.norm(cop-center[:2])),
       minimum_reference_gap_spanning_weight_line_m=float(gap),outer_nodes_eligible_at_spanning_gap=int((P[:,2]-plane<=gap).sum()),
       scope='Geometric lower bound at the fixed recorded reference pose, with arbitrary nonnegative normal reactions. No local compliance or pressure limit imposed.')
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':assess(*(Path(s).resolve() for s in sys.argv[1:]))
