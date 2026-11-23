"""Close each affected material cavity at measured boundary-edge fans."""
import numpy as np

def repair_cut(t,role,zone,mat,bad):
    target=np.flatnonzero(role==mat);cells=t[target];selected=zone[target].copy()
    limit=int(t.max())+1
    ei=np.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    def edge_keys(e):
        return e.min(-1).astype('int64')*limit+e.max(-1)
    keys=edge_keys(cells[:,ei])
    failed=np.isin(target,bad)
    info=[]
    for cycle in range(12):
        a=cells[selected];f=a[:,[(1,2,3),(0,3,2),(0,1,3),(0,2,1)]].reshape(-1,3)
        u,counts=np.unique(np.sort(f,axis=1),axis=0,return_counts=True)
        if counts.max()>2:raise ValueError('Nonmanifold faces in source cavity')
        ext=u[counts==1];edges,num=np.unique(edge_keys(ext[:,[(0,1),(1,2),(2,0)]]).ravel(),return_counts=True)
        bad_edges=edges[num!=2]
        if not len(bad_edges):break
        involved=np.isin(keys,bad_edges)
        missing=np.unique(keys[involved&~selected[:,None]])
        inherited=np.setdiff1d(bad_edges,missing)
        add=np.isin(keys,missing).any(1)&~selected
        trim=np.isin(keys,inherited).any(1)&selected
        necessary=np.unique(keys[failed][np.isin(keys[failed],inherited)])
        if len(necessary):trim=np.isin(keys,np.setdiff1d(inherited,necessary)).any(1)&selected
        info.append(dict(cycle=cycle,invalid_edges=len(bad_edges),extended=int(add.sum()),trimmed=int(trim.sum()),
                         inherited_incident_failed_edges=len(necessary)))
        if not np.any(add|trim):break
        selected[add]=True;selected[trim]=False
        if not np.all(selected[failed]):raise ValueError('Discarded an inverted source witness')
    return target[selected],info
