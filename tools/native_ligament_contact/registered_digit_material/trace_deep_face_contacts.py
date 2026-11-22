"""Recover strictly transverse edge/face witnesses using exact arithmetic on the exported physical coordinates."""
import csv
import json
from fractions import Fraction
from pathlib import Path
import sys
import numpy as np
from material_context import sha

def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def piercings(ids,xyz):
    witnesses=[]
    for direction in (0,1):
        front=ids[direction];back=ids[1-direction]
        a,b,c=(xyz[i] for i in back)
        normal=cross(sub(b,a),sub(c,a))
        if not dot(normal,normal):raise ValueError('Degenerate face')
        for k in range(3):
            n,m=front[k],front[(k+1)%3]
            p,q=xyz[n],xyz[m]
            u,v=dot(sub(p,a),normal),dot(sub(q,a),normal)
            if u*v>=0:continue
            t=u/(u-v)
            pos=tuple(i+t*(j-i) for i,j in zip(p,q))
            area=[dot(cross(sub(j,i),sub(pos,i)),normal) for i,j in ((a,b),(b,c),(c,a))]
            if min(area)<=0:continue
            witnesses.append(dict(edge=[int(n),int(m)],opposite_face=list(map(int,back)),t=float(t),
                minor_endpoint_plane_distance_nm=float(min(abs(u),abs(v)))/float(dot(normal,normal))**.5*1e9))
    return witnesses

def trace(run):
    cfg=json.loads((run/'launch.json').read_text())
    endpoint=json.loads((run/'verified-endpoint.json').read_text())
    layers=json.loads((run/'accepted-layers.json').read_text())
    if len(layers['witnesses'].get('2-2',[]))!=layers['pair_counts'][2][2]:
        raise ValueError('Incomplete deep witnesses')
    env=cfg['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def file(key,pin):
        path=Path(env[prefix+key])
        if sha(path)!=env[prefix+pin]:raise ValueError('Changed anatomical mapping')
        return path
    facefile=file('SURFACE_D_FACES_CSV','SURFACE_D_FACES_SHA256');nodefile=file('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    f=np.loadtxt(facefile,skiprows=1,usecols=(1,2,3),delimiter=',',dtype=int)
    pairs=layers['witnesses']['2-2'];depth_vertices=set(f[np.ravel(pairs)].ravel());node={}
    with nodefile.open() as stream:
        for r in csv.DictReader(stream):
            if r['registered_surface_layer']!='D':continue
            row=int(r['registered_surface_vertex_row'])
            if row in depth_vertices:
                if row in node or r['source_partition']!='fat':raise ValueError('Ambiguous deep-fat owner')
                node[row]=int(r['joint_node_id'])-1
    def coords(path):
        chosen=set(node.values());data={}
        with path.open() as stream:
            next(stream)
            for line in stream:
                r=line.split();i=int(r[0])-1
                if i in chosen:data[i]=tuple(Fraction.from_float(float(q)) for q in r[1:4])
        if chosen!=set(data):raise ValueError('Missing material point')
        return data
    installed=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE']);end=run/'physical-accepted.node'
    if (endpoint['physical_nodes_sha256']!=sha(end) or layers['inputs']['NATIVE_ACCEPTED_NODE_SHA256']!=sha(end)
            or cfg['pins'][str(installed)]!=sha(installed)):
        raise ValueError('Physical state differs from the audited snapshot')
    start,stop=coords(installed),coords(end)
    rows=[]
    for pair in pairs:
        ids=[[node[i] for i in f[t]] for t in pair]
        rows.append(dict(deep_faces=pair,material_nodes=ids,shared_nodes=sorted(set(ids[0])&set(ids[1])),
            installed_piercings=piercings(ids,start),accepted_piercings=piercings(ids,stop)))
    return dict(accepted_step=endpoint['accepted_steps'],frame_sha256=endpoint['frame_sha256'],installed_sha256=sha(installed),
        physical_position_sha256=sha(end),facets_sha256=sha(facefile),mapping_sha256=sha(nodefile),rows=rows,
        scope='Strict edge crossings through the interior of the opposite triangle, using exact binary64 values from the compensated native material export. Purely coplanar intersections require the original full triangle predicate.')

if __name__=='__main__':print(json.dumps(trace(Path(sys.argv[1]).resolve()),indent=2))
