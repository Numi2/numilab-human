"""Resolve new material surface rows on the connected historical tissue sheet."""
import json,sys
from collections import deque
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from compile_connected_joint import digest
from classify_connected_boundary import pack
from compile_material_boundary_lineage import closest
from inspect_shoulder_source import nearest

WEIGHTS=np.vstack((np.eye(3),np.full(3,1/3),[[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]]))

def edge_graph(faces):
    n=len(faces);edge=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    a=np.sort(edge.astype('i8'),axis=1);key=(a[:,0]<<32)|a[:,1];o=np.argsort(key,kind='stable')
    key=key[o];first=np.r_[0,1+np.flatnonzero(key[1:]!=key[:-1])];cnt=np.diff(np.r_[first,len(o)])
    if max(cnt)>2:raise ValueError('Nonmanifold role source sheet')
    p=first[cnt==2];u,v=o[p]%n,o[p+1]%n
    return coo_matrix((np.ones(len(p)*2,'u1'),(np.r_[u,v],np.r_[v,u])),shape=(n,n)).tocsr()

def run(h,job,lib,old,out):
    bound=np.load(job/'boundaries/faces.npz');source=np.load(old/'boundary-registration.npz');history=json.loads((old/'result.json').read_bytes())
    if history['mapping']!=digest(old/'boundary-registration.npz'):raise ValueError('Wrong material ancestry')
    env=json.loads((job/'posed-app/scene/launch.json').read_bytes())['environment'];pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    xyz=np.loadtxt(h/'closed-native/native/mesh.node',skiprows=1,usecols=(1,2,3));maps=np.load(h/'scene-registration/node-map.npz');table={};report={}
    for role in 'OID':
        path=lambda tag:Path(env[pr+'SURFACE_'+role+'_'+tag+'_CSV'])
        for tag in ['NODES','FACES']:
            if digest(path(tag))!=env[pr+'SURFACE_'+role+'_'+tag+'_SHA256']:raise ValueError('Historical surface hash mismatch')
        X=np.loadtxt(path('NODES'),delimiter=',',skiprows=1,usecols=(2,3,4));old_f=np.loadtxt(path('FACES'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
        surf=source[role+'_material_face'];orig=bound['ancestor'][surf];nold=history['layers'][role]['inherited']
        reorder=np.r_[np.argsort(orig[:nold]),np.arange(nold,len(surf))]
        known=orig[:nold][reorder[:nold]];new_faces=surf[nold:];FN=bound['face'][surf[reorder]]
        joint=maps[role];needed=np.unique(FN);is_used=np.zeros(len(xyz),bool);is_used[needed]=True
        valid=np.flatnonzero((joint>=0)&is_used[np.maximum(joint,0)])
        original=np.full(len(xyz),-1,'i4');original[joint[valid]]=valid;new_node=needed[original[needed]<0];N=np.r_[joint[valid],new_node]
        if len(N)!=len(needed) or len(np.unique(N))!=len(N):raise ValueError('Duplicated material node')
        lookup=np.full(len(xyz),-1,'i4');lookup[N]=np.arange(len(N))
        if not np.array_equal(pack(joint[old_f[known]]),pack(FN[:nold])):raise ValueError('Retained source connectivity changed')
        # Old face identities and orientation survive all compaction. Source roles may
        # orient the face oppositely to the positive tetrahedron convention.
        keys=FN[:nold];target=joint[old_f[known]];cyc=sum(np.all(keys==np.roll(target,j,axis=1),axis=1) for j in range(3))
        if not (np.all(cyc==1) or np.all(cyc==0)):raise ValueError('Inconsistent material layer orientation')
        fn=FN.copy()
        if np.all(cyc==0):fn[:,[1,2]]=fn[:,[2,1]]
        fn[:nold]=target
        table[role+'_node_joint']=N;table[role+'_face_nodes']=lookup[fn];table[role+'_global_facet']=surf[reorder];table[role+'_retained_source_node']=valid
        table[role+'_retained_source_face']=known
        current_node_geometry=np.max(abs(X[valid]-xyz[joint[valid]])) if len(valid) else 0
        row=dict(nodes=len(N),faces=len(FN),new_nodes=len(new_node),retained_faces=len(known),max_reference_displacement_of_retained_nodes_m=float(current_node_geometry))
        if not len(new_faces):report[role]=row;continue
        graph=edge_graph(old_f);first=graph.indptr;adj=graph.indices
        retired=np.ones(len(old_f),bool);retired[known]=False
        # Permit the actual changed patch and its first shared-edge rim. This prevents
        # nearest points across a fold or a disconnected anatomical region.
        allowed=retired.copy();allowed[graph@retired.astype('i4')>0]=True
        coordinates=(WEIGHTS@xyz[FN[nold:]]).reshape(-1,3)
        targets,dd,global_ids=nearest((X,old_f),coordinates,lib);dd=dd.reshape(-1,7);global_ids=global_ids.reshape(-1,7)
        previous=source[role+'_sample_source_face'];prior_gap=source[role+'_sample_gap_m'];offs=source[role+'_ancestry_offsets'];direct=source[role+'_volume_overlap_source_faces']
        ids=previous.copy();bary=source[role+'_sample_barycentric'].copy();gaps=prior_gap.copy();hops=np.zeros_like(ids);failure=[]
        for i,cell in enumerate(new_faces):
            seeds=set(direct[offs[i]:offs[i+1]].tolist())|set(previous[i].tolist());seeds.discard(-1)
            belt={s:0 for s in seeds};queue=deque(sorted(seeds))
            while queue:
                face=queue.popleft()
                for f in adj[first[face]:first[face+1]]:
                    if allowed[f] and f not in belt:belt[int(f)]=belt[face]+1;queue.append(int(f))
            for k in range(7):
                # Keep exact original identities on numerical ties.
                if gaps[i,k]<=dd[i,k]+2e-8:continue
                nearest_id=int(global_ids[i,k])
                if nearest_id not in belt:
                    failure.append([int(cell),k,float(gaps[i,k]),float(dd[i,k]),nearest_id]);continue
                point=coordinates[7*i+k];_,weights,d=closest(X[old_f[nearest_id:nearest_id+1]],point)
                if abs(d-dd[i,k])>2e-8:raise ValueError('Triangle reconstruction mismatch')
                ids[i,k]=nearest_id;bary[i,k]=weights;gaps[i,k]=d;hops[i,k]=belt[nearest_id]
        # A node has a single source embedding independent of how many new faces touch it.
        records={}
        for i,tri in enumerate(FN[nold:]):
            for k,v in enumerate(tri):
                if original[v]>=0:continue
                rec=(gaps[i,k],int(ids[i,k]),i,k)
                if v not in records or rec<records[v]:records[v]=rec
        if set(records)!=set(new_node):raise ValueError('New surface node lacks a material parent')
        owner=np.empty(len(new_node),'i4');node_bc=np.empty((len(new_node),3));res=np.empty_like(node_bc)
        for i,n in enumerate(new_node):
            d,f,j,k=records[n];owner[i]=f;node_bc[i]=bary[j,k];res[i]=xyz[n]-node_bc[i]@X[old_f[f]]
            if abs(np.linalg.norm(res[i])-d)>2e-8:raise ValueError('Nonconservative point reconstruction')
        table[role+'_new_node_parent_face']=owner;table[role+'_new_node_barycentric']=node_bc;table[role+'_new_node_residual_m']=res
        table[role+'_new_face_sample_source']=ids;table[role+'_new_face_sample_barycentric']=bary;table[role+'_new_face_sample_gap_m']=gaps
        row.update(edge_certified_improvements=int((ids!=previous).sum()),max_additional_edge_hops=int(hops.max()),maximum_sample_residual_m=float(gaps.max()),maximum_new_node_residual_m=float(np.max(np.linalg.norm(res,axis=1))),unresolved=sorted(failure,key=lambda x:-x[2])[:20],unresolved_count=len(failure))
        report[role]=row
    out.mkdir();np.savez_compressed(out/'registered-boundary.npz',**table)
    result=dict(source=digest(old/'boundary-registration.npz'),physical_mesh=history['reference'],layers=report,mapping=digest(out/'registered-boundary.npz'))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:]))
