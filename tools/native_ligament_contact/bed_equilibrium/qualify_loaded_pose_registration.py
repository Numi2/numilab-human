"""Verify physical donor identities and the registered deep attachments at a loaded posture."""
import csv,json,sys
from pathlib import Path
import numpy as np
from screen_registered_dense_posture import load_geometry,measure
from construct_registered_donor_field import errors
from evaluate_registered_bed_transfer import digest
from untangle_registered_donor_field import differentials
from recover_material_thickness import audit

def anchors(env):
    k='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def file(n,h):
        p=Path(env[k+n]);assert digest(p)==env[k+h];return p
    req=file('DEEP_ATTACHMENT_REQUESTS_CSV','DEEP_ATTACHMENT_REQUESTS_SHA256')
    mp=file('COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    lookup={}
    with req.open() as f:
        for a in csv.DictReader(f):
            key=(a['source_domain'],int(a['donor_body']),int(a['stable_source_vertex']))
            if key in lookup:raise ValueError('repeated donor attachment')
            lookup[key]=int(a['stable_attachment_id'])
    out={}
    with mp.open() as f:
        for row in csv.DictReader(f):
            if row['registered_surface_layer']!='D' or int(row['source_face_row'])>=0:continue
            key=(row['source_domain'],int(row['donor_body']),int(row['source_vertex_row']))
            if key in lookup:
                if key in out:raise ValueError('ambiguous attachment identity')
                out[key]=(lookup[key],int(row['joint_node_id'])-1)
    if set(out)!=set(lookup):raise ValueError(('unbound attachment',set(lookup)-set(out)))
    return out

def main(full,donor,pose):
    env=json.loads((full/'launch.json').read_text())['environment']
    bindings=anchors(env);x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    r=json.loads((pose/'result.json').read_text());assert r['source']==pins
    y=x+np.load(pose/'displacement.npy');q=y.astype('f4').astype(float)
    if not np.array_equal(q,y):raise ValueError('physical position is not Float32-exact')
    rest=x.astype('f4').astype(float);uhi=(q-rest).astype('f4').astype(float);ulo=(q-rest-uhi).astype('f4').astype(float)
    native=rest+uhi+ulo
    if not np.array_equal(native,q):raise ValueError('compensated material and Float32 world disagree')
    rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    j,sh,d,bad=audit(native,t,rr);assert not bad.any()
    records=[]
    for key,(a,n) in bindings.items():
        if owner[n]!=key[1]:raise ValueError('donor identity mismatch')
        records.append(dict(stable_attachment_id=a,volume_row=n,donor=key[1],target_m=target[n].tolist(),posed_m=y[n].tolist(),
                            position_residual_m=float(np.linalg.norm(y[n]-target[n]))))
    result=dict(source=pins,coordinate_encodings_identical=True,material=measure(native,owner,t,mat),
                minimum_relative_volume=float(j.min()),maximum_relative_volume=float(j.max()),
                min_outer_height_m=float(y[outer,2].min()),outer_rows_below_plane=int((y[outer,2]<0).sum()),
                attachment_rows=records,largest_attachment_error_m=max(v['position_residual_m'] for v in records),
                node_count=len(x),cell_count=len(t),surface_rows=int(outer.sum()))
    (pose/'loaded-registration.json').write_text(json.dumps(result,indent=2)+'\n')
    print('all cells pass both encodings; attachments',len(records),'largest residual m',result['largest_attachment_error_m'], 'skin below plane m',result['min_outer_height_m'])
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
