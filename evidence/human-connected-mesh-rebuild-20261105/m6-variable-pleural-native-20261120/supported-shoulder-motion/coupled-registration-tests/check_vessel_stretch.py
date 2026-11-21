"""Assess the existing vascular material-parent map in the independently compiled shoulder reference."""
from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

def check(launch,rigid,author,output):
    spec=importlib.util.spec_from_file_location("vessel_registration_owner",author); owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    cfg=json.loads(launch.read_text()); argv=cfg['argv']; receipt=Path(argv[argv.index('--resting-anatomy-receipt')+1])
    doc=json.loads(receipt.read_text());bd=doc['passive_tissue_parent_bindings']; src=bd['source_payloads'][7]; carrier=owner.parent_source_path(receipt,src)
    assert owner.sha_file(carrier)==src['sha256']
    nha=Path(doc['payload']['path']); assert owner.sha_file(nha)==bd['anatomy_payload_sha256']==doc['payload']['sha256']
    _,_,rows=owner.nha_codec._parse_payload(nha.read_bytes())
    data=owner.parent_owner._read_nhtiss4(carrier)
    _,before=owner.parent_owner._rest_body_poses_f32(Path(argv[1]));_,after=owner.parent_owner._rest_body_poses_f32(rigid)
    old,arrays=owner.row_maps_world(data,before);new,_=owner.row_maps_world(data,after)
    results=[]; materials={}
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
        raw=np.asarray(row['vertices6'][:,:3],dtype='<f4')
        _,first,weld=np.unique(raw,axis=0,return_index=True,return_inverse=True)
        seam_error=float(np.linalg.norm(target-target[first][weld],axis=1).max())
        # Use only original face edges. Duplicate seam coordinates never
        # create a shortcut in this certificate.
        first=weld=np.arange(len(raw))
        topo=np.unique(np.sort(weld[edges],axis=1),axis=0)
        topo=topo[topo[:,0]!=topo[:,1]]
        u,v=topo.T; ref=original[first]; dest=target[first]
        w=np.linalg.norm(ref[u].astype(float)-ref[v].astype(float),axis=1)
        graph=csr_matrix((np.tile(w,2),(np.r_[u,v],np.r_[v,u])),shape=(len(ref),len(ref)))
        lower=0.; pair=None
        arm_weight=np.zeros(len(original),float)
        for i,terms in enumerate(mapping['vertices']):
            for si,psid,slot,c in terms:
                assert si==7
                row_info=arrays[psid]
                for lane,w in zip(row_info['binding_indices'][slot],row_info['weights'][slot]):
                    if w:
                        arm_weight[i]+=c*w*(row_info['bindings'][lane]['core']==91)
        proximal=np.flatnonzero(arm_weight[first]<1e-10)
        distal=np.flatnonzero(arm_weight[first]>1-1e-10)
        assert len(proximal) and len(distal), (sid,arm_weight.min(),arm_weight.max(),list(arrays[1]))
        dist=dijkstra(graph,directed=False,indices=proximal)[:,distal]
        assert np.any(np.isfinite(dist))
        materials.update({f'p{sid}':ref[proximal],f'q{sid}':ref[distal],f'L{sid}':dist})
        for indices in np.array_split(proximal,max(1,len(proximal)//64)):
            geo=dijkstra(graph,directed=False,indices=indices)
            geo=geo[:,distal]
            dist=cdist(dest[indices],dest[distal])
            factor=np.divide(dist,geo,out=np.zeros_like(dist),where=geo>0)
            i,j=np.unravel_index(np.argmax(factor),factor.shape)
            if factor[i,j]>lower:
                lower=float(factor[i,j])
                pair=[int(first[indices[i]]),int(first[distal[j]]),float(geo[i,j]),float(dist[i,j])]
        results.append(dict(stable_id=sid,body=body,vertices=len(original),faces=len(faces),original_residual=residual,
            new_stale_residual=owner.error_stats(target,stale),edge_ratio=np.quantile(stretch,[0,.05,.5,.95,1]).tolist(),
            fully_fixed_vertex_counts=[len(proximal),len(distal)],registered_seam_residual_m=seam_error,minimum_possible_peak_edge_stretch=lower,material_pair_source_vertices_path_and_chord_m=pair,area_ratio=np.quantile(ar,[0,.05,.5,.95,1]).tolist(),
            max_material_point_motion_m=float(np.linalg.norm(target-original,axis=1).max())))
    np.savez_compressed(output.with_suffix('.npz'),**materials)
    output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':check(*map(Path,sys.argv[1:]))
