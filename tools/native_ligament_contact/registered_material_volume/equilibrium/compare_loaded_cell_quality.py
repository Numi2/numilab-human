"""Separate source, registration and native dynamic contributions to the same element shape metric."""
import json,sys
from pathlib import Path
import numpy as np
from supported_material_admission import geom
from compile_connected_joint import digest
def run(volume,case,published,path):
    rec=json.loads(published.read_bytes());a=case/'physical-accepted.node'
    files=[volume/'mesh.node',volume/'current.node',a]
    for pin,f in zip(('reference','loaded','physical_frame'),files):
        if digest(f)!=rec[pin]:raise ValueError('Unmatched '+pin)
    xyz=[np.loadtxt(f,skiprows=1,usecols=(1,2,3)) for f in files]
    raw=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');labels=np.loadtxt(volume/'current-child.labels.csv',skiprows=1,delimiter=',',dtype=np.int64)
    if not np.array_equal(labels[:,0],np.arange(len(raw))) or not np.array_equal(labels[:,1],raw[:,4]):raise ValueError('Unmatched material or cell ids')
    metrics={};floor=1.e-5
    for role,m in [('skin',15),('fat',16)]:
        idx=np.flatnonzero(raw[:,4]==m);t=raw[idx,:4]-1
        q=np.column_stack([geom(x,t)[1] for x in xyz]);bad=q<floor
        if np.any(bad[:,0]) or not np.array_equal(idx[bad[:,2]],rec['roles'][role]['failing_cells']):raise ValueError('Run differs from importer or native endpoint audit')
        ids=np.flatnonzero(bad[:,1:].any(1))
        result=dict(source_failures=int(sum(bad[:,0])),loaded_before_step=int(sum(bad[:,1])),after_first_step=int(sum(bad[:,2])),new_during_step=int(sum(~bad[:,1]&bad[:,2])),disappeared_during_step=int(sum(bad[:,1]&~bad[:,2])))
        result['minimum_shape']={k:float(np.min(q[:,i])) for i,k in enumerate(('source','loaded','accepted'))}
        result['cells']=[dict(row=int(idx[i]),region=labels[idx[i],2:5].tolist(),quality=q[i].tolist(),max_node_motion_m=float(np.linalg.norm(xyz[2][t[i]]-xyz[1][t[i]],axis=1).max()))for i in ids]
        metrics[role]=result
    path.write_text(json.dumps(dict(motion_sha256=digest(published),minimum_shape=floor,metrics=metrics),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
