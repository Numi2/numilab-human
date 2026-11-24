"""Resolve a native backtracking witness through the registered tissue and skeleton."""
import csv,json
from pathlib import Path
import sys
from bind_gluteal_receivers import digest


def locate(case,volume,output):
    with (case/'physical-posterior-tissue-objects.csv').open() as f:objects=list(csv.DictReader(f))
    index=next(i for i,r in enumerate(objects) if int(r['skin_fat']))
    address=sum(int(r['total_cells']) for r in objects[:index])
    trials=json.load(open('swept-comparison.json'))['traced-second-root']['last_geometric_witness'][0]['trials']
    global_ids={t['first_bad_tet'] for t in trials if t['reason']}
    labels=volume/'current-child.labels.csv'
    chosen={}
    with labels.open() as f:
        for row in csv.DictReader(f):
            i=int(row['child_tet_row'])+address
            if i in global_ids:
                chosen[i]=dict(global_element=i,local_element=i-address,material=int(row['material_index']),
                               donor=int(row['geometric_donor_body']),source_face=int(row['stable_source_face_id']))
            if len(chosen)==len(global_ids):break
    if set(chosen)!=global_ids:raise ValueError('Witness outside registered skin/fat owner')
    manifest=Path(json.loads((case/'launch.json').read_bytes())['argv'][3]).with_suffix('.manifest.json')
    def collect(x,names):
        if isinstance(x,dict):
            if 'core_body_index' in x and 'myosim_body' in x:
                names.setdefault(x['core_body_index'],set()).add(x['myosim_body'])
            for item in x.values():collect(item,names)
        elif isinstance(x,list):
            for item in x:collect(item,names)
    names={}
    collect(json.loads(manifest.read_bytes()),names)
    for r in chosen.values():
        n=names[r['donor']]
        if len(n)!=1:raise ValueError('Ambiguous donor')
        r['donor_name']=next(iter(n))
    output.write_text(json.dumps(dict(labels=digest(labels),skeleton=digest(manifest),global_element_offset=address,
                                      source_objects=list(chosen.values())),indent=2)+'\n')


if __name__=='__main__':locate(*(Path(s) for s in sys.argv[1:]))
