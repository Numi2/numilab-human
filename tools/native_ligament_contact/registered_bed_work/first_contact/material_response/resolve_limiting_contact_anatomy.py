"""Resolve the captured minimum contact to its registered layer and actual topological neighborhood."""
import json,re,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(job,volume,fit,trial,out):
    logs=(trial/'stderr.log').read_text().splitlines()
    capture=next(json.loads(s.split('=',1)[1]) for s in logs if s.startswith('fem_deformable_limit_trace='))
    record=min(capture['contacts'],key=lambda r:r['lane_bound_after'])
    if record['lane_bound_after']!=capture['line_search_alpha'][0]:raise ValueError('Reported minimum does not match the applied bound')
    obj,offset,count=map(int,re.search(r'object=(\d+).*state_offset=(\d+) state_count=(\d+)',next(s for s in logs if s.startswith('registered_material_span '))).groups())
    if record['identity'][:3]!=[obj,obj,0]:raise ValueError('Expected vertex-face self-contact')
    nodes=np.array([k-offset for k in record['nodes'] if k!=0xffffffff])
    if len(nodes)!=4 or min(nodes)<0 or max(nodes)>=count:raise ValueError('Wrong volume state span')
    b=np.load(job/'registered-boundary/registered-boundary.npz');layers=[s for s in 'OID' if np.isin(nodes,b[s+'_node_joint']).all()]
    if len(layers)!=1:raise ValueError('Ambiguous physical layer')
    layer=layers[0];idx=b[layer+'_node_joint'];faces=idx[b[layer+'_face_nodes']]
    target=np.flatnonzero((np.isin(faces,nodes[1:])).all(1))
    if len(target)!=1:raise ValueError('Ambiguous physical facet')
    rows=[int(np.flatnonzero(idx==n)[0])for n in nodes]
    source=b[layer+'_retained_source_node'];target_id=int(target[0])
    if max(rows)>=len(source) or target_id>=len(b[layer+'_retained_source_face']):raise ValueError('Requires embedded source-ancestry lookup')
    coords=lambda path:np.loadtxt(path,usecols=(1,2,3),skiprows=1).astype('f4').astype(float)[nodes]
    x=coords(volume/'mesh.node');y=coords(fit/'current.node')
    adjacent=[np.flatnonzero((faces==nodes[0]).any(1)&(faces==n).any(1)).tolist() for n in nodes[1:]]
    report=dict(launch=digest(trial/'launch.json'),contact_slot=record['contact_slot'],layer=layer,
        nodes=nodes.tolist(),retained_source_nodes=source[rows].tolist(),face=target_id,
        retained_source_face=int(b[layer+'_retained_source_face'][target_id]),shared_mesh_edge_faces=adjacent,
        source_vertex_to_face_vertices_m=np.linalg.norm(x[1:]-x[0],axis=1).tolist(),
        registered_vertex_to_face_vertices_m=np.linalg.norm(y[1:]-y[0],axis=1).tolist(),
        contact_parameters=record,selected_alpha=capture['line_search_alpha'][0],
        scope='The vertex neighbors one corner of the triangle. It has a distinct identity from its closest corner. Source adjacency alone does not establish that this pair can be excluded.')
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
