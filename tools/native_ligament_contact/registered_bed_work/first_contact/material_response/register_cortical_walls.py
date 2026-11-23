"""Bind the physical cortical shoulder boundary to incident native cells and source bone triangles."""
from collections import defaultdict
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from inspect_shoulder_source import nearest
from compile_connected_joint import digest

def run(closed,plc,domain,volume,conflicts,bones,lib,out):
    src=np.load(closed);prepared=np.load(plc);z=np.load(domain)
    ancestry=np.load(volume/'material-source-indices.npz');origin=int(ancestry['child_begin'])
    receipt=json.loads((volume/'result.json').read_text());bone_record=json.loads((conflicts/'result.json').read_bytes())
    if digest(domain)!=receipt['domain_sha256'] or digest(bones)!=bone_record['bones'] or digest(volume/'mesh.ele')!=receipt['output_hashes']['mesh.ele']:raise ValueError('Wrong current domain')
    w=np.load(conflicts/'witnesses.npz')
    dst,perm=cKDTree(prepared['vertex_world_m']).query(src['vertex_world_m'],k=2)
    if dst[:,0].max()>1e-8 or (dst[:,1]<=dst[:,0]).any() or len(np.unique(perm[:,0]))!=len(perm):raise ValueError('Unresolved input vertex identity')
    parents={};edges=defaultdict(set)
    faces=perm[:,0][src['faces']]
    for j in np.where(np.isin(src['marker'],(115,116)))[0]:
        t=sorted(faces[j]);key=(int(src['marker'][j]),*t)
        if key in parents: raise ValueError('Duplicate original wall')
        parents[key]=j
        for a,b in [(t[0],t[1]),(t[1],t[2]),(t[0],t[2])]:edges[(key[0],a,b)].add(j)
    t=z['t'];global_cells=np.loadtxt(volume/'mesh.ele',skiprows=1+origin,dtype='i4');g=global_cells[:,1:5]-1
    if len(t)!=len(g) or not np.array_equal(global_cells[:,5],z['role']) or not np.array_equal(global_cells[:,0],np.arange(origin+1,origin+1+len(t))):raise ValueError('Native cell order differs')
    mapping=np.full(len(z['x']),-1,'i4');mapping[t.ravel()]=g.ravel()
    if not np.array_equal(mapping[t],g):raise ValueError('Native node identity differs')
    uses=defaultdict(list)
    for side,corners in enumerate([(1,2,3),(0,3,2),(0,1,3),(0,2,1)]):
        for j,row in enumerate(t[:,corners]):uses[tuple(sorted(row))].append((j,side,row))
    rows=[];seeds=[];offsets=[0];exact=[];centers=[]
    for face,marker in zip(z['facets'],z['markers']):
        if marker not in (115,116):continue
        a,b,c=map(int,sorted(face));unit=uses[a,b,c]
        if len(unit)!=1:raise ValueError('Internal face presented as cortical boundary')
        i,side,oriented=unit[0];key=(int(marker),a,b,c)
        if z['role'][i]+100!=marker:raise ValueError('Cortical wall material mismatch')
        direct=parents.get(key);lineage=[direct] if direct is not None else sorted(set.union(*(edges[k] for k in [(int(marker),a,b),(int(marker),a,c),(int(marker),b,c)])))
        banks=np.unique(src['cortex_bank'][lineage])
        if len(banks)!=1 or int(banks[0]) not in range(8,14):raise ValueError('Ambiguous cortical owner')
        rows.append([origin+i,side,int(marker)-100,int(banks[0]),*map(int,mapping[oriented])])
        exact.append(direct is not None);seeds.extend(np.unique(src['parent_face'][lineage]).tolist());offsets.append(len(seeds));centers.append(z['x'][oriented].astype(float))
    rows=np.array(rows,dtype='i4');tri=np.array(centers);sites=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])@tri;hits=np.empty((len(rows),3),'i4');bary=np.empty((len(rows),3,3));gaps=np.empty((len(rows),3));counts={}
    for b in np.unique(rows[:,3]):
        select=rows[:,3]==b;v=w['cortex'+str(b)];f=w['faces'+str(b)];q=sites[select].reshape(-1,3)
        p,d,ids=nearest((v,f),q,lib)
        corners=v[f[ids]].astype(float);base=corners[:,0];A=np.swapaxes(corners[:,1:]-base[:,None],1,2);u=np.einsum('nij,nj->ni',np.linalg.pinv(A),p-base)
        weights=np.column_stack((1-u.sum(1),u));rebuild=np.einsum('ni,nij->nj',weights,corners)
        if not np.isfinite(weights).all() or weights.min()<-2e-7 or np.max(np.linalg.norm(rebuild-p,axis=1))>2e-8:raise ValueError('Inconsistent source triangle coordinates')
        hits[select]=ids.reshape(-1,3);bary[select]=weights.reshape(-1,3,3);gaps[select]=d.reshape(-1,3);counts[int(b)]=dict(facets=int(select.sum()),max_gap_m=float(d.max()))
    if len(set(map(tuple,rows[:,:2])))!=len(rows):raise ValueError('Multiple physical owners for the same wall')
    out.mkdir();np.savez_compressed(out/'registered-cortical-facets.npz',cell_side_material_bank_nodes=rows,exact_source_topology=np.array(exact),candidate_face_offsets=np.array(offsets),candidate_source_faces=np.array(seeds),quadrature_barycentric=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]]),cortex_faces=hits,cortex_barycentric=bary,reference_distance_m=gaps)
    report=dict(domain=digest(domain),nodes=receipt['output_hashes']['mesh.node'],elements=receipt['output_hashes']['mesh.ele'],bones=digest(bones),rigid=bone_record['rigid'],cortex_source=digest(conflicts/'witnesses.npz'),cortical_facet_map=digest(out/'registered-cortical-facets.npz'),facets=len(rows),exact_parent_faces=sum(exact),retriangulated=len(rows)-sum(exact),quadrature_points=3*len(rows),per_bank=counts)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(counts)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
