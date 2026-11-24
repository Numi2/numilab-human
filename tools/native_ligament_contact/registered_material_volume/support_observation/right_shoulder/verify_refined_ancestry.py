"""Independently check emitted labels against P1 maxima and their source face identities."""
import json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def verify(w,vol,binding,output):
    z=np.load(vol/'source-overlay.npz');targets=z['geometric_child'];source=z['geometric_parent']
    original=np.loadtxt(w/'material-import/volume/current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    labels=np.loadtxt(vol/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    selected=np.zeros(len(labels),dtype=bool);selected[targets]=True
    if not np.array_equal(labels[:len(original)][~selected[:len(original)]],original[~selected[:len(original)]]):
        raise ValueError('Modified unrelated source labels')
    if not np.array_equal(labels[targets,1:4],original[source,1:4]):raise ValueError('Changed tissue, regional identity or mechanical owner')
    p1=binding/'carrier-replay/material-p1';info=json.loads((p1/'result.json').read_bytes())
    for role,identity in [('skin',15),('fat',16)]:
        g=np.load(p1/(role+'-role-geometry.npz'));idx=g['combined_child_rows']
        p=np.load(info['environment']['NUMI_H706_FRACTIONAL_P1_'+role.upper()+'_NPZ'])
        p={k:p[k] for k in ('cell_group_offsets','cell_group_p1_score_numerators','cell_group_p1_score_denominators','cell_group_ids','group_tuples','cell_group_stable_face_offsets','cell_group_stable_face_rows')}
        for j in np.flatnonzero(selected[idx]):
            pos=np.arange(*p['cell_group_offsets'][j:j+2])
            sc=p['cell_group_p1_score_numerators'][pos]/p['cell_group_p1_score_denominators'][pos]
            tag=labels[idx[j],1:];valid=False
            for slot in pos[sc==sc.max()]:
                gid=p['cell_group_ids'][slot]
                if not np.array_equal(p['group_tuples'][gid],tag[:3]):continue
                begin,end=p['cell_group_stable_face_offsets'][slot:slot+2]
                faces=p['cell_group_stable_face_rows'][begin:end]
                valid|=tag[3] in faces or len(np.unique(faces))>1 and tag[3]==0x80000000+int(gid)
            if not valid:raise ValueError(f'No matched P1 source face at row {idx[j]}')
    mat=json.loads((vol/'native-import.json').read_bytes())
    mass=json.loads((binding/'carrier-replay/native-replay/result.json').read_bytes())
    if mat['label_sha256']!=digest(vol/'current-child.labels.csv') or mass['exit']!=0:
        raise ValueError('Final native import or source allocation failed')
    r=dict(final_native=mat,source_mass_replay=mass['receipts'],verified_cells=len(targets),reclassified_stable_source_faces=int((labels[targets,4]!=original[source,4]).sum()),unmatched_faces=0,changed_material_or_donor=0,patch=digest(vol/'source-overlay.npz'),native_allocation=digest(binding/'carrier-replay/native-replay/result.json'))
    output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':verify(*(Path(a).resolve()for a in sys.argv[1:]))
