"""Carry the registered material sidecars through a certified local volume remesh."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from rebuild_current_material_basis import lineage


def run(work):
    q=work.parent
    v=q/'qualified-material'
    prior=q.parent/'relative/volume'
    out=work/'binding/tables'
    out.mkdir(parents=True,exist_ok=True)
    env=json.loads((q/'gravity-scene/launch.json').read_bytes())['environment']
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    pins={}
    stats=Counter()
    def source(k):
        p=Path(env[prefix+k+'_CSV'])
        if digest(p)!=env[prefix+k+'_SHA256']:raise ValueError(k+' input digest')
        return p
    def register(k,f):
        pins[prefix+k+'_CSV']=str(f)
        pins[prefix+k+'_SHA256']=digest(f)
    def mesh(path):return np.loadtxt(path,skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    def points(path):return np.loadtxt(path,skiprows=1,usecols=(1,2,3))
    def reverse(kept,size):
        result=np.full(size,-1,'i4')
        result[kept]=np.arange(len(kept))
        return result
    def triples(x):
        a=np.ascontiguousarray(np.sort(x,axis=1),dtype='<i4')
        return a.view('V12').ravel()
    fmt=lambda x:format(float(x),'.17g')
    x=points(v/'mesh.node')
    prev=points(prior/'mesh.node')
    raw=mesh(v/'mesh.ele')
    tet=raw[:,:4]-1
    mat=raw[:,4]
    old=mesh(prior/'mesh.ele')
    z=np.load(q/'seam-volume/local-ancestry.npz')
    node=z['surviving_source_nodes']
    rn=reverse(node,len(prev))
    weld=rn.copy()
    op=np.load(q/'obstacle-seam/volume-and-facets.npz')
    weld[op['retired_vertices']]=rn[op['surviving_vertices']]
    if np.min(weld)<0:raise ValueError('Unknown weld ancestry')
    parents=lineage(q)
    remap=np.full(len(old),-1,'i4')
    kept=np.flatnonzero(parents>=0)
    remap[parents[kept]]=kept
    exact=(tet[kept]==rn[old[parents[kept],:4]-1]).all(1)
    changed=np.union1d(kept[~exact],np.flatnonzero(parents<0))
    vertices=np.unique(np.r_[tet[changed].ravel(),weld[old[remap<0,:4].ravel()-1]])
    local=np.flatnonzero(np.isin(tet,vertices).any(1))
    incident=defaultdict(list)
    face_owner=defaultdict(list)
    for c in local:
        for i,n in enumerate(tet[c]):
            incident[n].append((int(c),i))
            face_owner[(mat[c],tuple(sorted(np.delete(tet[c],i))))].append(int(c))
    changedold=set(parents[kept[~exact]].tolist())
    dreg=np.load(q.parent.parent/'registered-boundary/registered-boundary.npz')
    D=dreg['D_node_joint']
    df=dreg['D_face_nodes']
    dn=rn[D]
    dr=reverse(np.flatnonzero(dn>=0),len(D))
    origfaces=z['D_original_face_rows']
    fr=reverse(origfaces,len(df))
    bound=weld[D[df]][origfaces]
    exactD=np.load(v/'material-walls.npz')['D']
    if not np.array_equal(np.sort(triples(bound)),np.sort(triples(exactD))):raise ValueError('Deep surface ancestry differs from emitted wall')
    survivingD=dn[dn>=0]
    dlookup=np.full(len(node),-1,'i4')
    dlookup[survivingD]=np.arange(len(survivingD))
    surface=dlookup[bound]
    if (surface<0).any():raise ValueError('Unbound material surface')
    # Recompute normals at the welded one-ring, preserving the recorded outward orientation.
    changed_d=np.unique(df[op['old_D_faces']])
    original_norms=np.loadtxt(source('SURFACE_D_NODES'),skiprows=1,delimiter=',',usecols=(5,6,7))
    detailed=x[survivingD]
    normal=np.zeros_like(detailed)
    area=np.cross(detailed[surface[:,1]]-detailed[surface[:,0]],detailed[surface[:,2]]-detailed[surface[:,0]])
    for i in range(3):np.add.at(normal,surface[:,i],area)
    normals={}
    for r in changed_d:
        k=dr[r]
        if k<0:continue
        length=np.linalg.norm(normal[k]);sign=np.sign(normal[k]@original_norms[r])
        if not (length>0 and sign):raise ValueError('No registered normal')
        normals[k]=list(map(fmt,normal[k]*(sign/length)))
    contact=q/'registered-material-contacts'
    thick=np.loadtxt(contact/'tables/D-thickness.csv',skiprows=1,delimiter=',',usecols=(1,2))
    tnodes=np.load(contact/'physical-owner/registered-boundary/registered-boundary.npz')['D_node_joint']
    if not np.array_equal(np.sort(survivingD),tnodes):raise ValueError('Contact thickness ownership')
    for key in ['SURFACE_D_NODES','ROLE_D_FULL86','ROLE_D_CONTACT_THICKNESS','SURFACE_D_FACES','NODE_REGISTRATION']:
        f=out/(key.lower()+'.csv')
        with source(key).open() as inp,f.open('w') as dst:
            dst.write(inp.readline())
            for idx,line in enumerate(inp):
                fields=line.rstrip('\r\n').split(',')
                if key=='NODE_REGISTRATION' and fields[0]!='D':
                    dst.write(line);continue
                col=int(key=='NODE_REGISTRATION'); i=int(fields[col])
                k=fr[i] if key=='SURFACE_D_FACES' else dr[i]
                if k<0:continue
                fields[col]=str(k)
                if key=='SURFACE_D_FACES':fields[1:4]=list(map(str,surface[k]))
                if key=='SURFACE_D_NODES' and k in normals:fields[5:8]=normals[k]
                if key=='ROLE_D_CONTACT_THICKNESS':fields[1:3]=list(map(fmt,thick[np.searchsorted(tnodes,survivingD[k])]))
                dst.write(','.join(fields)+'\n')
        register(key,f)
    frames={}
    with source('DONOR_FRAMES').open() as f:
        for r in csv.DictReader(f):
            frames[int(r['donor'])]=np.array([float(r[f'current_R{i}{j}']) for i in range(3)for j in range(3)]).reshape(3,3)
    labels=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')
    moved=set(np.flatnonzero((x!=prev[node]).any(1)).tolist())
    f=out/'combined-node-map.csv'
    with source('COMBINED_NODE_MAP').open()as inp,f.open('w')as dst:
        dst.write(inp.readline())
        for line in inp:
            r=line.rstrip('\r\n').split(',')
            oldn=int(r[0])-1;n=rn[oldn]
            if n<0:stats['retired nodes']+=1;continue
            r[0]=str(n+1)
            if r[6]=='D':r[7]=str(dr[int(r[7])])
            c=int(r[43]);k=remap[c]
            if k<0 or c in changedold:
                choices=[(a,b) for a,b in incident[n] if mat[a]==old[c,4]]
                if not choices or np.any(np.array(r[48:52],float)):raise ValueError('Unmapped source node '+str((oldn,c)))
                k,b=min(choices,key=lambda ab:(tuple(labels[ab[0],2:4])!=(int(r[40]),int(r[41])),ab))
                r[40:42]=list(map(str,labels[k,2:4]))
                r[44:48]=[str(int(j==b))for j in range(4)]
                stats['rebound operational parents']+=1
            r[43]=str(k)
            if n in moved:
                if r[6] or r[18]!='none':raise ValueError('Moved authored boundary node')
                delta=x[n]-np.array(r[9:12],float)
                r[9:12]=list(map(fmt,x[n]))
                r[14:17]=list(map(fmt,np.array(r[14:17],float)+np.linalg.solve(frames[int(r[12])],delta)))
                residual=np.array(r[34:37],float)+delta
                r[34:38]=list(map(fmt,[*residual,np.linalg.norm(residual)]))
                stats['updated interior coordinates']+=1
            dst.write(','.join(r)+'\n')
    register('COMBINED_NODE_MAP',f)
    # Current role row numbers and immutable historical source rows have separate columns.
    original_deep_map={tuple(sorted(tri)):i for i,tri in enumerate(D[df])}
    f=out/'facet-ancestry.csv'
    with source('FACET_ANCESTRY').open()as inp,f.open('w')as dst:
        rd=csv.reader(inp);wr=csv.writer(dst,lineterminator='\n');wr.writerow(next(rd))
        for r in rd:
            c=int(r[18]);k=remap[c];nodes=[int(v)-1 for v in r[19:22]]
            tri=tuple(int(weld[n])for n in nodes)
            if r[3]=='inner_D':
                i=original_deep_map[tuple(sorted(nodes))]
                if fr[i]<0:stats['retired D facets']+=1;continue
                # Historical face_source_row values use the original source, independently
                # from the geometric shell face; preserve that ancestry.
            if k<0 or c in changedold:
                cand=face_owner[(old[c,4],tuple(sorted(tri)))]
                if len(cand)!=1:raise ValueError('Ambiguous exposed face '+str((r[3],nodes,cand)))
                k=cand[0]
                stats['rebound surface owners']+=1
            if int(r[17])!=c+1:raise ValueError('Inconsistent registered element')
            r[17:22]=[k+1,k,*[n+1 for n in tri]]
            if 'headwall' in r[3] and r[16]=='[-1,-1,-1]':
                if json.loads(r[23])!=[c]:raise ValueError('Unmapped headwall')
                r[23]=json.dumps([int(k)])
            wr.writerow(r)
    register('FACET_ANCESTRY',f)
    if stats['retired D facets']!=len(df)-len(surface) or stats['updated interior coordinates']!=len(moved):raise ValueError('Incomplete mapping')
    (out/'result.json').write_text(json.dumps(dict(environment=pins,stats=stats),indent=2)+'\n')

if __name__=='__main__':run(Path(__file__).resolve().parent)
