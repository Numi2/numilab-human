"""Assess the existing vascular material-parent map in the independently compiled shoulder reference."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np

def check(launch,rigid,author,output):
    spec=importlib.util.spec_from_file_location("vessel_registration_owner",author); owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    cfg=json.loads(launch.read_text()); argv=cfg['argv']; receipt=Path(argv[argv.index('--resting-anatomy-receipt')+1])
    doc=json.loads(receipt.read_text());bd=doc['passive_tissue_parent_bindings']; src=bd['source_payloads'][7]; carrier=owner.parent_source_path(receipt,src)
    assert owner.sha_file(carrier)==src['sha256']
    nha=Path(doc['payload']['path']); assert owner.sha_file(nha)==bd['anatomy_payload_sha256']==doc['payload']['sha256']
    _,_,rows=owner.nha_codec._parse_payload(nha.read_bytes())
    data=owner.parent_owner._read_nhtiss4(carrier)
    _,before=owner.parent_owner._rest_body_poses_f32(Path(argv[1]));_,after=owner.parent_owner._rest_body_poses_f32(rigid)
    old,_=owner.row_maps_world(data,before);new,_=owner.row_maps_world(data,after)
    results=[]
    for mapping in bd['anatomy_surfaces']:
        sid=int(mapping['stable_id']);row=rows[sid]; body=int(row['body_index'])
        original,_=owner.map_world(mapping,old,7); target,_=owner.map_world(mapping,new,7)
        original_child=owner.child_world(row,before[body]); stale=owner.child_world(row,after[body])
        residual=owner.error_stats(original,original_child);assert residual['count_over_2um']==0
        faces=np.asarray(row['faces']); t0=original[faces].astype(float);t1=target[faces].astype(float)
        n0=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);n1=np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
        ar=np.linalg.norm(n1,axis=1)/np.linalg.norm(n0,axis=1)
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
        a,b=edges.T
        stretch=np.linalg.norm(target[a]-target[b],axis=1)/np.linalg.norm(original[a]-original[b],axis=1)
        assert np.isfinite(stretch).all() and np.isfinite(ar).all()
        results.append(dict(stable_id=sid,body=body,vertices=len(original),faces=len(faces),original_residual=residual,
            new_stale_residual=owner.error_stats(target,stale),edge_ratio=np.quantile(stretch,[0,.05,.5,.95,1]).tolist(),
            area_ratio=np.quantile(ar,[0,.05,.5,.95,1]).tolist(),
            max_material_point_motion_m=float(np.linalg.norm(target-original,axis=1).max())))
    output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':check(*map(Path,sys.argv[1:]))
