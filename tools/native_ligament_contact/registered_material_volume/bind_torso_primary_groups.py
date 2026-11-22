"""Bind moved fine-cell numerical recipient labels to their recomputed exact P1 maximum."""
import csv,json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
base,reg,proposal,output=map(Path,sys.argv[1:]);p1=reg/'material/p1';p=np.load(proposal)
old=base/'paired-floor-child.labels.csv';assert sha(old)==json.load((base/'paired-floor-wall-lineage.json').open())['labels_sha256']
labels=np.loadtxt(old,delimiter=',',skiprows=1,dtype='i8');original=labels.copy()
audit=[]
for role in ('skin','fat'):
    r=json.load((p1/(role+'-p1-receipt.json')).open());support=Path(r['p1_path']);assert sha(support)==r['p1_sha256']
    s=np.load(support);geom=np.load(r['role_geometry_path']);ids=geom['combined_child_rows']
    tu=s['group_tuples'];g=s['cell_group_ids'];scores=s['cell_group_p1_score_numerators'];off=s['cell_group_offsets'];idx=np.repeat(np.arange(len(ids)),np.diff(off));best=np.maximum.reduceat(scores,off[:-1])
    maximum=scores==best[idx];same=np.all(tu[g]==original[ids[idx],1:4],axis=1);bad=np.flatnonzero(~np.logical_or.reduceat(maximum&same,off[:-1]))
    if not np.isin(ids[bad],p['cell']).all():raise ValueError('The repair alters an unrelated cell registration')
    for i in bad:
        cell=ids[i];opts=np.arange(off[i],off[i+1]);opts=opts[maximum[opts]]
        # Preserve the registered geometric contributor across a tie whenever
        # possible, then the recorded region. Stable source tuple order breaks
        # the remaining ambiguity reproducibly.
        choice=min(opts,key=lambda j:(tu[g[j],1]!=original[cell,2],tu[g[j],2]!=original[cell,3],int(g[j])))
        tup=tu[g[choice]]
        assert tup[0]==original[cell,1]
        labels[cell,2:4]=tup[1:3]
        audit.append((cell,*original[cell,2:4],*labels[cell,2:4],len(opts),int(scores[choice]),int(s['cell_group_p1_score_denominators'][choice])))
assert np.array_equal(labels[:,[0,1,4]],original[:,[0,1,4]])
with output.open('w') as f:
    with old.open() as src:f.write(next(src))
    np.savetxt(f,labels,fmt='%d',delimiter=',')
with output.with_suffix('.changes.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['cell','previous_donor','previous_region','current_donor','current_region','number_of_maxima','p1_numerator','p1_denominator']);w.writerows(audit)
report=dict(previous_sha256=sha(old),labels_sha256=sha(output),native_p1_state_sha256=sha(p1/'current-paired-volume-state.json'),source_proposal_sha256=sha(proposal),changed_inferred_cell_primary_groups=len(audit),material_and_source_face_changes=0,mechanical_donor_node_changes=0,changed_outside_cavity=0,ties=sum(x[5]>1 for x in audit))
output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
