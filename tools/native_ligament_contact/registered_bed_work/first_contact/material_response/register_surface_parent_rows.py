"""Preserve registered parent identity and update metric residuals for a repaired skin/fat surface."""
import csv
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from register_material_surfaces import PREFIX,pinned,sha,vector,write_vector


def run(launch, layers, dest):
    env=json.loads(launch.read_bytes())['environment']
    receipt=json.loads((layers/'registration.json').read_bytes())
    updates={}
    counts={}
    for k in 'OID':
        old=pinned(env,'SURFACE_'+k+'_NODES_CSV','SURFACE_'+k+'_NODES_SHA256')
        new=layers/old.name
        if sha(new)!=receipt['layers'][k]['emitted']:
            raise ValueError('Changed material surface')
        counts[k]=receipt['layers'][k]['rows']
        with old.open() as a,new.open() as b:
            for x,y in zip(csv.DictReader(a),csv.DictReader(b),strict=True):
                if all(x['w'+c]==y['w'+c] for c in 'xyz'):continue
                if x['vertex']!=y['vertex']:raise ValueError('Changed surface identity')
                updates[k,int(y['vertex'])]=(vector(y,'w')-vector(x,'w'),vector(y,'source_residual_','_m'))
    origin=pinned(env,'NODE_REGISTRATION_CSV','NODE_REGISTRATION_SHA256')
    limit=float(env[PREFIX+'MAX_PARENT_REGISTRATION_DISTANCE_M'])
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage=Path(tmp)/'parent'
        stage.mkdir()
        target=stage/'node-parent-registration.csv'
        seen={k:np.zeros(n,bool) for k,n in counts.items()}
        largest=0.
        moved=0
        with origin.open() as f,target.open('w') as out:
            reader=csv.DictReader(f)
            writer=csv.DictWriter(out,reader.fieldnames,lineterminator='\n')
            writer.writeheader()
            for row in reader:
                k,i=row['layer'],int(row['surface_vertex_row'])
                if i<0 or seen[k][i]:raise ValueError('Duplicate or invalid parent row')
                seen[k][i]=True
                if (k,i) in updates:
                    delta,residual=updates[k,i]
                    write_vector(row,'source_residual_',residual,'_m')
                    if row['parent_tet_row'] not in ('','-1'):
                        q=vector(row,'parent_residual_','_m')+delta
                        distance=float(np.linalg.norm(q))
                        if not np.isfinite(distance) or distance>limit:raise ValueError(f'Parent distance exceeds {limit} at {k} {i}')
                        write_vector(row,'parent_residual_',q,'_m')
                        row['parent_registration_distance_m']=format(distance,'.17g')
                        largest=max(largest,distance)
                    moved+=1
                writer.writerow(row)
        if moved!=len(updates) or not all(v.all() for v in seen.values()):
            raise ValueError('Incomplete surface-parent registration')
        (stage/'registration.json').write_text(json.dumps(dict(source=sha(origin), target=sha(target),
              layers=sha(layers/'registration.json'),rows=sum(counts.values()),changed=moved,
              maximum_changed_parent_distance_m=largest,limit_m=limit),indent=2)+'\n')
        if dest.exists():raise FileExistsError(dest)
        stage.rename(dest)


if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
