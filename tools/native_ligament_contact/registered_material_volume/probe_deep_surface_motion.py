"""Identify indexed features and the native displacement in deep-boundary crossings."""
import csv,sys,json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from repair_accepted_material_cells import coords
from emit_skin_source_registration import sha
def gather(d):
    a=json.loads((d/'launch.json').read_text())['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    surf=Path(a[prefix+'SURFACE_D_FACES_CSV']);assert sha(surf)==a[prefix+'SURFACE_D_FACES_SHA256']
    combined=Path(a[prefix+'COMBINED_NODE_MAP_CSV']);assert sha(combined)==a[prefix+'COMBINED_NODE_MAP_SHA256']
    t=np.loadtxt(surf,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    ids=np.full(int(t.max())+1,-1,dtype='i4')
    with combined.open() as stream:
        for row in csv.DictReader(stream):
            if row['registered_surface_layer']=='D':
                k=int(row['registered_surface_vertex_row']);assert ids[k]==-1;ids[k]=int(row['joint_node_id'])-1
    assert ids.min()>=0
    f=ids[t];a=json.loads((d/'accepted-layers.json').read_text())
    rest=coords(d/'endpoint-reference.node');end=coords(d/'endpoint-accepted.node')
    assert sha(d/'endpoint-accepted.node')==a['inputs']['NATIVE_ACCEPTED_NODE_SHA256']
    if sha(d/'endpoint-reference.node')!=a['inputs']['NATIVE_REFERENCE_NODE_SHA256'] or not np.array_equal(rest,coords(d/'mesh/mesh.node')):
        raise ValueError('The endpoint has a different material reference')
    pairs=np.array(a['witnesses']['2-2'],dtype='u4')
    tri=f[pairs].reshape(len(pairs),6)
    graph=np.zeros((len(pairs),len(pairs)),dtype=bool)
    for i,row in enumerate(tri):graph[i]=np.isin(tri,row).any(1)
    n,labels=connected_components(coo_matrix(graph))
    return f,rest,end,pairs,labels
if __name__=='__main__':
    d=Path(sys.argv[1]);f,rest,end,pairs,g=gather(d);reg=[]
    for k in np.unique(g):
        rows=pairs[g==k];ns=np.unique(f[rows]);disp=end[ns]-rest[ns]
        rec=dict(count=int(len(rows)),nodes=ns.tolist(),pairs=rows.tolist(),reference_center_m=rest[ns].mean(0).tolist(),
            max_relative_motion_m=float(np.linalg.norm(disp-disp.mean(0),axis=1).max()),shared_nodes=[int(np.intersect1d(f[a],f[b]).size) for a,b in rows],
            representative_face_nodes=f[rows[0]].tolist())
        reg.append(rec)
        print(len(rows),len(ns),np.round(rest[ns].mean(0),3),np.round(rec['max_relative_motion_m']*1e6,5),rec['shared_nodes'])
    Path(sys.argv[2]).write_text(json.dumps(reg,indent=2)+'\n')
