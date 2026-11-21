"""Diagnose an accepted active pressure basis without changing a force or an admission gate."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_context import sha
a,z,out=map(Path,sys.argv[1:]);d=json.loads((a/'manifest.json').read_text());m=d['motion_contact_registry']
pool=np.fromfile(a/'pool-material-points.bin','f4').reshape(-1,7,4)
owners=np.fromfile(a/'motion-cortical-owners.bin','u4').reshape(-1,4);threshold=d['finite_facets']['start']+16*m['original_pairs']
mapped={int(r['slot']):int(r['cortical_row']) for r in csv.DictReader((z/'accepted-cortical-registry.csv').open())}
rows=list(csv.DictReader((z/'accepted-pelvic-state.csv').open()))
def field(k):return {int(r['index']):np.array([r[q] for q in 'xyzw'],'f4').astype(float) for r in rows if r['field']==k}
weights=field('reverse_shape');parents=field('reverse_parent_nodes');sites=field('normal_and_force')
loc=field('material_node');free=np.fromfile(a/'free-node-rows.bin','u4').reshape(-1,4)[:,2]!=0
outdoc={'capture':sha(z/'accepted-pelvic-state.csv'),'step':int(rows[0]['accepted_step']),'sides':{}}
for side in range(2):
    columns=[];loads=[]
    for k,v in sites.items():
        if v[3]<=0:continue
        if k<threshold:
            q=pool[k,3].view('u4');s=int(q[1]==145);b=int(q[2])
        else:
            _,b,s,_=owners[mapped[(k-threshold)//16]]
        if s!=side:continue
        n=parents[k].astype(int) if k in parents else pool[k,5].view('u4');w=weights.get(k,pool[k,6])
        col=np.zeros((len(loc),3));np.add.at(col,n,w[:,None]*v[:3]);columns.append(col[free].ravel());loads.append(float(v[3]))
    if not columns:continue
    A=np.column_stack(columns);A=A[np.any(A!=0,axis=1)]
    if not len(A):continue
    # SVD of the small contact Gram matrix is used only to count resolvable
    # force directions. Square roots below the FP64 Gram noise floor are excluded.
    s=np.sqrt(np.maximum(0.,np.linalg.eigvalsh(A.T@A)))[::-1];n=len(s)
    ranks={str(t):int(sum(s>t*s[0])) for t in [1e-7,1e-6,1e-5,1e-4]}
    outdoc['sides'][str(side)]={'loaded_rows':n,'active_free_vector_components':A.shape[0],
        'nonzero_columns_on_free_nodes':int(np.count_nonzero(np.linalg.norm(A,axis=0))),
        'singular_value_max':float(s[0]),'rank_relative_threshold':ranks,
        'pressure_magnitude_sum_N':sum(loads),'free_nodal_force_l2_N':float(np.linalg.norm(A@loads))}
out.write_text(json.dumps(outdoc,indent=2)+'\n')
