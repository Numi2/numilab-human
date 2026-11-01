import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).with_name('inner_endpoint_quotient_002.py')
s=importlib.util.spec_from_file_location('endpoint_quotient_002_test',P)
q=importlib.util.module_from_spec(s);s.loader.exec_module(q)


def fixture_volume():
    p=np.asarray(((0.,0.,0.),(1.e-7,0.,0.),(0.,1.,0.),(0.,0.,1.),
                  (-1.,-1.,-1.),(1.,-1.,-1.)),np.float32)
    t=np.asarray(((0,1,2,3),(2,0,3,4),(1,2,3,5)),np.int64)
    t=q.M.orient(p,t)
    b=q.boundary_faces(t)
    return {'points_world_m':p,'points_world_m_f32':p.copy(),'tetrahedra':t,
            'material_indices':np.full(len(t),16,np.int32),'desired_boundary_faces':b.copy(),
            'boundary_faces':b.copy()}


def fixture_surface(v):
    f=q.boundary_faces(v['tetrahedra'])
    n=len(f);p=v['points_world_m']
    off=np.arange(n+1,dtype=np.int64)
    source={
       'points_world_m':p.astype(np.float64),'points_world_m_f32':p.copy(),'faces':f,
       'quality_star_point_output_to_input_row':np.arange(len(p),dtype=np.int64),
       'quality_star_face_output_to_input_row':np.arange(n,dtype=np.int64),
       'source_input_face_rows':np.arange(n,dtype=np.int64),
       'source_input_vertex_quotient':np.arange(len(p),dtype=np.int64),
       'quality_star_patch_union_offsets':off,
       'face_ancestry_offsets':off,
       'source_collapse_operation_rows':np.full(n,-1,np.int64),
       'typed_parent_candidate_path':np.asarray(['parent.npz']),
       'typed_parent_candidate_sha256':np.asarray(['a'*64]),
    }
    for lane in q.LANES:
        source['face_source_'+lane]=np.full(n,1,np.int64)
        source['quality_star_patch_union_face_source_'+lane]=np.full(n,1,np.int64)
    for lane in q.ANCESTRY_LANES:
        source['face_ancestry_'+lane]=np.arange(n,dtype=np.int64)
    source['face_source_identity_schema_by_original_id_json']=np.asarray(['{}'])
    return source


def assert_reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('negative control unexpectedly passed')


def run():
    v=fixture_volume();s=fixture_surface(v)
    drop,keep=0,1
    # D is one-sided to Fat; I is an atomic paired operation on Skin and Fat.
    d=q.propose_endpoint_quotient('D',s,{'Fat':v},(0,1),drop=drop,keep=keep)
    assert d['volumes']['Fat']['native_bad_after']==0
    assert len(d['volumes']['Fat']['tetrahedra'])==2
    i=q.propose_endpoint_quotient('I',s,{'Skin':fixture_volume(),'Fat':fixture_volume()},(0,1),drop=drop,keep=keep)
    assert i['paired'] and i['combined_native_bad_after']==0
    sr=q.collapse_surface(s['points_world_m_f32'],s['faces'],(0,1),drop=drop,keep=keep)
    so=q.endpoint_surface_arrays(s,sr,parent_path='parent.npz',parent_sha='a'*64,operation_id=0)
    assert so['points_world_m'].tobytes()==s['points_world_m'].tobytes()
    assert so['points_world_m_f32'].tobytes()==s['points_world_m_f32'].tobytes()
    assert len(so['faces'])==len(sr['faces'])
    assert len(so['quality_star_patch_union_offsets'])==len(so['faces'])+1
    assert len(so['face_ancestry_offsets'])==len(so['faces'])+1
    for lane in q.LANES:
        assert len(so['face_source_'+lane])==len(so['faces'])
        assert so['quality_star_patch_union_offsets'][-1]==len(so['quality_star_patch_union_face_source_'+lane])
    vr=q.collapse_volume(q.M,v['points_world_m'],v['tetrahedra'],v['material_indices'],v['desired_boundary_faces'],s['points_world_m_f32'],(0,1),drop=drop,keep=keep)
    vo=q.endpoint_volume_arrays(v,vr,parent_path='parent-volume.npz',parent_sha='b'*64)
    assert vo['points_world_m'].tobytes()==v['points_world_m'].tobytes()
    assert vo['points_world_m_f32'].tobytes()==v['points_world_m_f32'].tobytes()
    assert q.face_chain(q.boundary_faces(vo['tetrahedra']))==q.face_chain(vo['desired_boundary_faces'])
    assert np.all(vo['input_node_to_output_node']==np.asarray((1,1,2,3,4,5)))
    assert_reject(lambda:q.collapse_surface(s['points_world_m_f32'],s['faces'],(0,1),drop=drop,keep=keep,fixed_endpoint_rows=(drop,)))
    assert_reject(lambda:q.collapse_surface(s['points_world_m_f32'],s['faces'],(0,1),drop=drop,keep=keep,max_edge_m=1.e-8))
    assert_reject(lambda:q.collapse_surface(s['points_world_m_f32'],s['faces'],(0,2),drop=0,keep=2))
    print({'status':'PASS_ENDPOINT_QUOTIENT_CONTROLS','D_one_volume':True,'I_paired_atomic':True,
           'point_arrays_byte_identical':True,'typed_union_and_ancestry_rows_aligned':True,
           'exact_boundary_quotient':True,'native_bad_tet_removed':True,
           'fixed_endpoint_rejected':True,'over_cap_rejected':True,'nonedge_rejected':True})

if __name__=='__main__':run()
