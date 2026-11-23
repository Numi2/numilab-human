"""Refresh only changed historical node parents, retaining the identical registered boundary."""
import csv,json,sys
from pathlib import Path
from compile_connected_joint import digest

def run(tables,previous,current,out):
    out.mkdir();doc=json.loads((tables/'result.json').read_bytes());updates={}
    for k in 'ID':
        with (previous/(k+'-new.csv')).open() as a,(current/(k+'-new.csv')).open() as b:
            for before,row in zip(csv.DictReader(a),csv.DictReader(b),strict=True):
                permitted={'parent_registration_kind','parent_tet_row','parent_registration_tie_count','parent_registration_distance_m','source_parent_geometric_donor_body','source_parent_anatomical_region_id'}
                for key in row:
                    if key in permitted or key.startswith(('parent_bary','parent_current_residual_')):continue
                    if before[key]!=row[key]:raise ValueError('Source geometry or physical semantics changed: '+key)
                updates[k,int(row['registered_surface_vertex_row'])]=row
    file=out/'node-parent-registration.csv';prior=doc['files']['NODE_REGISTRATION'];seen=set()
    if digest(Path(prior['path']))!=prior['sha256']:raise ValueError('Parent table differs')
    with Path(prior['path']).open() as inp,file.open('x') as dest:
        head=inp.readline();dest.write(head);keys=head.strip().split(',')
        for line in inp:
            cols=line.rstrip().split(',');k=cols[0],int(cols[1])
            if k not in updates:dest.write(line);continue
            r=updates[k];seen.add(k)
            for i,name in enumerate(keys):
                if name.startswith('parent_'):cols[i]=r[name.replace('parent_residual_','parent_current_residual_')]
            dest.write(','.join(cols)+'\n')
    if seen!=set(updates):raise ValueError('Missing observed source rows')
    for key,path in [('COMBINED_NODE_MAP',current/'combined-node-map.csv'),('NODE_REGISTRATION',file)]:
        doc['files'][key]=dict(path=str(path),sha256=digest(path))
    doc['updated_parents']=len(seen);(out/'result.json').write_text(json.dumps(doc,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
