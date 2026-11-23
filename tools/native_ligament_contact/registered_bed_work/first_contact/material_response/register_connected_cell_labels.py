"""Bind a rebuilt tetrahedron's primary donor and face to the maxima of its integrated P1 field."""
import json,sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned,sha

def run(launch,output):
    env=json.loads(launch.read_bytes())['environment'];pre='NUMI_H706_FRACTIONAL_'
    p=pinned(env,'CHILD_LABELS','CHILD_LABELS_SHA256');a=np.loadtxt(p,delimiter=',',skiprows=1,dtype='i8')
    with p.open() as f:header=f.readline().strip()
    if not np.array_equal(a[:,0],np.arange(len(a))):raise ValueError('Invalid input row identity')
    src=np.loadtxt(pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),delimiter=',',skiprows=1,dtype='i8')
    output.mkdir();result={}
    for tag,m in [('SKIN',15),('FAT',16)]:
        f=Path(env[pre+'P1_'+tag+'_NPZ'])
        if sha(f)!=env[pre+'P1_'+tag+'_SHA256']:raise ValueError('Wrong P1 domain')
        d=dict(np.load(f));ids=np.flatnonzero(a[:,1]==m);o=d['cell_group_offsets'];g=d['cell_group_ids'];tup=d['group_tuples']
        score=d['cell_group_p1_score_numerators'].astype('f8')/d['cell_group_p1_score_denominators'];n=len(ids)
        if np.max(abs(np.add.reduceat(score,o[:-1])-1))>1e-12:raise ValueError('P1 partition of unity failed')
        maximum=np.maximum.reduceat(score,o[:-1]);local=np.repeat(np.arange(n),np.diff(o));win=score==maximum[local];match=np.all(tup[g]==a[ids[local],1:4],axis=1);ok=np.zeros(n,bool);ok[local[win&match]]=True
        affected=np.flatnonzero(~ok)
        witnesses=[];ties=0
        for c in affected:
            slots=np.flatnonzero(win[o[c]:o[c+1]])+o[c];ties+=int(len(slots)>1)
            # Prefer previously authored cortical-source face in an equally weighted group.
            eligible=[]
            for z in slots:
                pp=d['cell_group_parent_tet_offsets'];v=d['cell_group_parent_tet_rows'][pp[z]:pp[z+1]]
                if not len(v) or not np.all(src[v,1:4]==tup[g[z]]):raise ValueError('Wrong parent ancestry')
                faces=np.unique(src[v,4]);oldface=a[ids[c],4];face=oldface if oldface in faces else faces[0]
                eligible.append((face!=oldface,int(g[z]),int(face)))
            _,group,face=min(eligible);j=ids[c];prior=a[j,2:5].copy();a[j,1:4]=tup[group];a[j,4]=face
            if len(witnesses)<6:witnesses.append(dict(child=int(j),old=prior.tolist(),new=a[j,2:5].tolist()))
        result[tag]=dict(cells=n,changed=len(affected),ties=ties,witnesses=witnesses)
    name=output/'current-child.labels.csv';np.savetxt(name,a,fmt='%d',delimiter=',',header=header,comments='')
    p='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_CHILD_LABELS'
    (output/'result.json').write_text(json.dumps(dict(roles=result,environment={p:str(name),p+'_SHA256':sha(name)}),indent=2)+'\n')
    print(result,flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
