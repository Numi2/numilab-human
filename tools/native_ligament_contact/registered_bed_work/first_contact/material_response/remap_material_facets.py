"""Resolve every registered shell facet to exactly one emitted tetrahedron of its material."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from registered_bed_reference import read,sha

def remap(parent,mesh,dest):
    env=read(parent/'native-terminal-16/launch.json')['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    source=Path(env[pre+'FACET_ANCESTRY_CSV'])
    if sha(source)!=env[pre+'FACET_ANCESTRY_SHA256']:raise ValueError('Unmatched facet source')
    n=read(mesh/'receipt.json')['nodes']
    def keys(vertices,mat):
        v=np.sort(vertices,axis=-1);return ((v[:,0]*n+v[:,1])*n+v[:,2])*2+mat-15
    oldT=np.loadtxt(env[pre+'TETGEN_ELEMENTS'],dtype=np.int32,skiprows=1,usecols=(1,2,3,4,5))
    target=np.loadtxt(mesh/'mesh.ele',dtype=np.int32,skiprows=1,usecols=(1,2,3,4,5))
    with source.open() as handle:
        rd=csv.DictReader(handle);cols=rd.fieldnames
        K=[];prev=[]
        for r in rd:
            local=[int(r[f'joint_node{i}'])-1 for i in range(3)];m={'skin':15,'fat':16}[r['source_partition']]
            row=int(r['child_tet_row'])
            if int(r['joint_element_id'])!=row+1 or int(oldT[row,4])!=m or not set(local)<=set(oldT[row,:4]-1):
                raise ValueError('Uncertified source facet')
            a,b,c=sorted(local);K.append(((a*n+b)*n+c)*2+m-15);prev.append(row)
    key=np.asarray(K,dtype='i8');order=np.argsort(key);sorted_keys=key[order]
    if np.any(np.diff(sorted_keys)==0):raise ValueError('Duplicate signed material facet')
    match=np.full(len(key),-1,np.int32);facets=np.asarray([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    for start in range(0,len(target),32768):
        C=target[start:start+32768];tri=(C[:,:4]-1)[:,facets].reshape(-1,3).astype('i8')
        query=keys(tri,np.repeat(C[:,4],4));i=np.searchsorted(sorted_keys,query);pos=np.flatnonzero(i<len(key))
        pos=pos[sorted_keys[i[pos]]==query[pos]];rid=order[i[pos]]
        if len(np.unique(rid))!=len(rid) or (match[rid]!=-1).any():raise ValueError('Multiple same-material incident cells on a registered facet')
        match[rid]=start+pos//4
    if np.any(match<0):raise ValueError(f'{sum(match<0)} lost registered material facets')
    dest.mkdir(exist_ok=True);out=dest/'facet-ancestry-current.csv'
    labels = np.loadtxt(mesh/'current-child.labels.csv',delimiter=',',skiprows=1,usecols=(4,),dtype='i8')
    corrected_walls=0
    with source.open()as fp,out.open('w') as outfp:
        reader=csv.DictReader(fp);writer=csv.DictWriter(outfp,cols,lineterminator='\n');writer.writeheader()
        for idx,r in enumerate(reader):
            target_cell=int(match[idx])
            if r['face_source_role'] in ('donor_skin_OI_headwall','donor_fat_ID_headwall') and r['parent_source_node_rows']=='[-1,-1,-1]':
                # The final field names the incident cell in the *current* joint mesh.
                # Its two source-face fields remain in their immutable atlas domain.
                origin=json.loads(r['face_ancestry_source_rows'])
                if (json.loads(r['face_ancestry_parent_face_rows']) != [int(prev[idx])] or
                    len(origin)!=1 or origin!=json.loads(r['face_ancestry_source_original_ids']) or
                    origin != [int(labels[target_cell])]):
                    raise ValueError('Derived wall no longer traces to the same source face')
                r['face_ancestry_parent_face_rows']=f'[{target_cell}]'
                corrected_walls+=target_cell!=prev[idx]
            r['joint_element_id']=str(target_cell+1)
            r['child_tet_row']=str(target_cell)
            writer.writerow(r)
    result=dict(source=sha(source),tetrahedra=sha(mesh/'mesh.ele'),recovered_facets=len(key),reindexed_derived_wall_rows=int(corrected_walls),missing=0,duplicate=0,changed_incident_row=int(np.count_nonzero(match!=prev)),result=sha(out))
    (dest/'facets.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':remap(*(Path(s).resolve()for s in sys.argv[1:]))
