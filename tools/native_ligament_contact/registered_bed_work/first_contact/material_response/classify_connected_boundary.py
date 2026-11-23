"""Identify every one-sided facet and skin/fat interface in the emitted native mesh."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
def pack(f):
    a=np.sort(np.asarray(f,dtype=np.int64),axis=1)
    if a.size and (a.min()<0 or a.max()>=1<<20):raise ValueError('Node identity outside exact key')
    return (a[:,0]<<40)|(a[:,1]<<20)|a[:,2]
def run(h,trial,out):
    out.mkdir();cfg=json.loads((trial/'launch.json').read_text())['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    file=Path(cfg[pre+'TETGEN_ELEMENTS']);tet=np.loadtxt(file,dtype=np.int32,skiprows=1);mat=tet[:,-1].copy()
    if not np.array_equal(tet[:,0],np.arange(len(tet))+1):raise ValueError('Unexpected TetGen numbering')
    t=tet[:,1:5]-1;n=len(t);K=np.empty(4*n,'i8')
    # Positive tetrahedron boundary orientation indexed by the omitted local corner.
    corners=[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]
    for side,ijk in enumerate(corners):K[side*n:(side+1)*n]=pack(t[:,ijk])
    ids=np.argsort(K,kind='stable').astype('i4');keys=K[ids];del K
    first=np.r_[0,1+np.flatnonzero(keys[1:]!=keys[:-1])];counts=np.diff(np.r_[first,len(keys)])
    if max(counts)>2:raise ValueError('Nonmanifold tetrahedral face')
    bd=first[counts==1];p=first[counts==2];a,b=ids[p]%n,ids[p+1]%n;ab=p[mat[a]!=mat[b]]
    src=np.r_[ids[bd],np.where(mat[ids[ab]%n]==15,ids[ab],ids[ab+1])]
    cells=src%n;side=src//n;F=t[cells[:,None],np.array(corners)[side]]
    keys=pack(F);order=np.argsort(keys);keys=keys[order];F=F[order];cells=cells[order];side=side[order]
    interface=(np.arange(len(F))>=len(bd))[order];code=np.where(interface,1,np.where(mat[cells]==15,0,2)).astype('i1')
    # Look up genuine historical authored wall faces, all inherited O/I/D, and
    # quadrature-certified new cortical faces. Never infer head closure from a hole.
    kin=np.load(h/'scene-registration/node-map.npz');parent=kin['source_to_emitted']
    mapped=np.full(len(F),-1,'i4');kind=np.full(len(F),-1,'i1')
    roles={};bad=[]
    for stage in 'OID':
        si=kin[stage];fpath=Path(cfg[pre+'SURFACE_'+stage+'_FACES_CSV']);old=np.loadtxt(fpath,skiprows=1,delimiter=',',usecols=(0,1,2,3),dtype='i4')
        v=si[old[:,1:]];ok=(v>=0).all(1);key=pack(v[ok]);i=np.searchsorted(keys,key);found=(i<len(keys))&(keys[np.minimum(i,len(keys)-1)]==key)
        idx=i[found];parent_rows=old[:,0][ok][found];k='OID'.index(stage)
        if len(np.unique(idx))!=len(idx) or np.any(kind[idx]>=0) or np.any(code[idx]!=k):raise ValueError('Registered facet owner mismatch '+stage)
        mapped[idx]=parent_rows;kind[idx]=k
        roles[stage]={'old_source_rows':len(old),'inherited':len(idx)}
    ancestry=Path(cfg[pre+'FACET_ANCESTRY_CSV']);wr=[];names={}
    with ancestry.open() as inp:
        for row in csv.DictReader(inp):
            name=row['face_source_role'];names[name]=names.get(name,0)+1
            if 'wall' in name:wr.append([int(row['joint_node'+str(k)])-1 for k in range(3)]+[int(row['shell_face_row'])])
    if not wr:raise ValueError('Missing historical head walls')
    wr=np.array(wr);v=parent[wr[:,:3]]
    if np.any(v<0):raise ValueError('Former head wall node removed')
    idx=np.searchsorted(keys,pack(v))
    if np.any(idx>=len(keys)) or not np.array_equal(keys[idx],pack(v)) or np.any(kind[idx]!=-1):raise ValueError('Historical head wall lost or reassigned')
    kind[idx]=3;mapped[idx]=wr[:,3]
    rr=np.load(h/'cortical-registration/registered-cortical-facets.npz')['cell_side_material_bank_nodes']
    k=pack(rr[:,-3:]);idx=np.searchsorted(keys,k)
    if np.any(idx>=len(keys)) or not np.array_equal(keys[idx],k) or np.any(kind[idx]!=-1) or np.any(interface[idx]) or np.any(cells[idx]!=rr[:,0]):raise ValueError('Cortical facet does not match physical child cell')
    kind[idx]=4;mapped[idx]=np.arange(len(rr))
    # All other changed faces must be supported by material incidence, ancestry
    # and conformity before attachment, thickness or contact is inherited.
    changed={x:int(np.sum((kind<0)&(code==i))) for i,x in enumerate('OID')}
    np.savez_compressed(out/'faces.npz',face=F,cell=cells,side=side,layer=code,inherited_kind=kind,ancestor=mapped)
    report=dict(elements=digest(file),node_map=digest(h/'scene-registration/node-map.npz'),cortical=digest(h/'cortical-registration/registered-cortical-facets.npz'),cells=n,boundary=len(bd),shared=len(ab),inherited_layers=roles,head_walls=len(wr),cortical_walls=len(rr),new_or_retriangulated=changed,initial_roles=names,face_table=digest(out/'faces.npz'))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:]))
