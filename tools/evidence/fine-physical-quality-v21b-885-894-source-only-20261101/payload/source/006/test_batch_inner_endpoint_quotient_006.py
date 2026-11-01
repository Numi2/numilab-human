import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
core=load('batch006_test_core',HERE/'inner_endpoint_quotient_002.py')
b=load('batch006_test_builder',HERE/'batch_inner_endpoint_quotient_006.py');b.core=core
fx=load('batch006_fixture',HERE/'test_inner_endpoint_quotient_002.py')

class IdentityRebinder:
    @staticmethod
    def rebind_source_aliases(role,arrays,surfaces,wallxyz,wallrows,wallids,*,input_pins=None):
        return arrays,{'status':'PASS_SYNTHETIC_IDENTITY_REBIND'}

def combine():
    v0=fx.fixture_volume();v1=fx.fixture_volume()
    v1['points_world_m']=v1['points_world_m']+np.asarray([0,0,10],np.float32)
    v1['points_world_m_f32']=v1['points_world_m'].copy()
    pts=np.vstack((v0['points_world_m'],v1['points_world_m']))
    t=np.vstack((v0['tetrahedra'],v1['tetrahedra']+6))
    mat=np.r_[v0['material_indices'],v1['material_indices']]
    faces=core.boundary_faces(t)
    vol={'points_world_m':pts.astype(np.float64),'points_world_m_f32':pts.copy(),'tetrahedra':t,
         'material_indices':mat,'desired_boundary_faces':faces.copy(),'boundary_faces':faces.copy()}
    n=len(faces);off=np.arange(n+1,dtype=np.int64)
    src={'points_world_m':pts.astype(np.float64),'points_world_m_f32':pts.copy(),'faces':faces.copy(),
         'quality_star_point_output_to_input_row':np.arange(len(pts),dtype=np.int64),
         'quality_star_face_output_to_input_row':np.arange(n,dtype=np.int64),
         'source_input_face_rows':np.arange(n,dtype=np.int64),
         'source_input_vertex_quotient':np.arange(len(pts),dtype=np.int64),
         'quality_star_patch_union_offsets':off.copy(),'face_ancestry_offsets':off.copy(),
         'source_collapse_operation_rows':np.full(n,-1,np.int64),
         'typed_parent_candidate_path':np.asarray(['parent.npz']),'typed_parent_candidate_sha256':np.asarray(['a'*64]),
         'endpoint_quotient_parent_path':np.asarray(['parent.npz']),'endpoint_quotient_parent_sha256':np.asarray(['a'*64])}
    for i,lane in enumerate(core.LANES):
        vals=np.arange(n,dtype=np.int64)+100*i
        src['face_source_'+lane]=vals.copy();src['quality_star_patch_union_face_source_'+lane]=vals.copy()
    for i,lane in enumerate(core.ANCESTRY_LANES):src['face_ancestry_'+lane]=np.arange(n,dtype=np.int64)+1000*i
    return vol,src

def run():
    vol,src=combine();basev={k:np.array(v,copy=True) for k,v in vol.items()};bases={k:np.array(v,copy=True) for k,v in src.items()}
    curv={'Fat':vol};curs={'D':src};smap=np.arange(len(src['points_world_m_f32']),dtype=np.int64);fmap=np.arange(len(src['faces']),dtype=np.int64)
    nmap={'Fat':np.arange(len(vol['points_world_m']),dtype=np.int64)};cmap={'Fat':np.arange(len(vol['tetrahedra']),dtype=np.int64)};unions={'Fat':{}};ops=[]
    for serial,(drop,keep) in enumerate(((0,1),(6,7)),1):
        edge=(drop,keep);got=core.propose_endpoint_quotient('D',curs['D'],{'Fat':curv['Fat']},edge,drop=drop,keep=keep)
        before=fmap.copy();sres=got['surface'];pr=np.asarray(sres['face_output_to_parent_rows'],np.int64)
        changedbase=before[pr[np.asarray(sres['changed_output_rows'],np.int64)]];starbase=before[np.asarray(sres['complete_operation_parent_face_rows'],np.int64)]
        smap[drop]=smap[keep];fmap=before[pr]
        vr=got['volumes']['Fat'];affected=cmap['Fat'][np.asarray(vr['affected_input_cell_rows'],np.int64)].tolist();keepcells=np.asarray(vr['output_cell_to_input_cell_rows'],np.int64)
        nu={j:set(unions['Fat'][int(old)]) for j,old in enumerate(keepcells) if int(old) in unions['Fat']}
        for j in np.asarray(vr['changed_output_cell_rows'],np.int64):nu.setdefault(int(j),set()).update(affected)
        ops.append({'operation_id':serial,'role':'D','edge_rows':[drop,keep],'drop_source_row':drop,'keep_source_row':keep,
            'changed_face_base_rows':changedbase.tolist(),'star_face_base_rows':starbase.tolist(),
            'affected_base_cells':{'Fat':affected},'volume_delta_det6_integer':{'Fat':str(vr['exact_signed_volume_delta_det6_integer'])},
            'drop_volume_rows':{'Fat':int(vr['drop_volume_row'])},'keep_volume_rows':{'Fat':int(vr['keep_volume_row'])}})
        curs={'D':core.endpoint_surface_arrays(curs['D'],sres,parent_path='parent.npz',parent_sha='a'*64,operation_id=serial)}
        curv={'Fat':core.endpoint_volume_arrays(curv['Fat'],vr,parent_path='parent-volume.npz',parent_sha='b'*64)}
        curv['Fat']['mapped_current_boundary_faces']=curv['Fat']['boundary_faces'].copy()
        cmap={'Fat':cmap['Fat'][keepcells]};unions={'Fat':nu}
        nmap['Fat'][int(vr['drop_volume_row'])]=nmap['Fat'][int(vr['keep_volume_row'])]
    sf=b.finalize_surface(curs['D'],bases,smap,fmap,ops,{'path':'parent.npz','sha256':'a'*64})
    vf=b.finalize_volume(curv['Fat'],basev,nmap['Fat'],cmap['Fat'],unions['Fat'],ops,{'path':'parent-volume.npz','sha256':'b'*64},'Fat')
    boundary=core.boundary_faces(vf['tetrahedra']);vf['boundary_faces']=boundary;vf['mapped_current_boundary_faces']=boundary.copy();vf['proposed_current_desired_boundary_faces']=vf['desired_boundary_faces'].copy()
    target=len(vf['desired_boundary_faces'])
    for k,dtype in (('desired_boundary_source_group',np.int8),('desired_boundary_source_rows',np.int64),('desired_boundary_source_face_rows',np.int64),
        ('proposed_current_desired_boundary_source_group',np.int8),('proposed_current_desired_boundary_source_rows',np.int64),('proposed_current_desired_boundary_source_face_rows',np.int64)):
        vf[k]=np.zeros(target,dtype=dtype)
    vf['fat_current_target_rebind_wall_faces_volume_rows']=np.empty((0,3),np.int64);vf['fat_current_target_rebind_wall_face_points_world_m_f32']=np.empty((0,3,3),np.float32)
    sc=b.verify_final_surface(bases,sf,smap,fmap,ops,{'path':'parent.npz','sha256':'a'*64})
    vc=b.verify_final_volume(basev,vf,nmap['Fat'],cmap['Fat'],unions['Fat'],ops,IdentityRebinder,'Fat',{'D':sf},{'path':'parent-volume.npz','sha256':'b'*64})
    assert sc['full_typed_union_and_ancestry_replayed'] and vc['full_replacement_parent_union_replayed']
    print({'status':'PASS_BATCH006_TWO_DISJOINT_OPERATIONS','two_edge_maps_composed':True,'source_face_ancestry_flattened':True,
           'cell_parent_unions_flattened':True,'full_final_replay':True,'native_bad_before_after':(vc['native_bad_before'],vc['native_bad_after'])})

if __name__=='__main__':run()
