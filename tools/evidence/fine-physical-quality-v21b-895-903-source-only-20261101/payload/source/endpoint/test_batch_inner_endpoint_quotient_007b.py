import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
core=load('batch007_test_core',HERE/'inner_endpoint_quotient_002.py')
b=load('batch007_test_builder',HERE/'batch_inner_endpoint_quotient_007b.py');b.core=core
b6=load('batch007_test_flattened',HERE/'batch_inner_endpoint_quotient_006.py');b6.core=core;b.b6=b6
fx=load('batch007_fixture',HERE/'test_inner_endpoint_quotient_002.py')

class IdentityRebinder:
    @staticmethod
    def rebind_source_aliases(role,arrays,surfaces,wallxyz,wallrows,wallids,*,input_pins=None):
        return arrays,{'status':'PASS_SYNTHETIC_IDENTITY_REBIND'}

def combine():
    v0=fx.fixture_volume();v1=fx.fixture_volume()
    v1['points_world_m']=v1['points_world_m']+np.asarray([0,0,10],np.float32);v1['points_world_m_f32']=v1['points_world_m'].copy()
    pts=np.vstack((v0['points_world_m'],v1['points_world_m']));t=np.vstack((v0['tetrahedra'],v1['tetrahedra']+6))
    faces=core.boundary_faces(t);mat=np.r_[v0['material_indices'],v1['material_indices']]
    vol={'points_world_m':pts.astype(np.float64),'points_world_m_f32':pts.copy(),'tetrahedra':t,'material_indices':mat,
         'desired_boundary_faces':faces.copy(),'boundary_faces':faces.copy()}
    n=len(faces);off=np.arange(n+1,dtype=np.int64)
    src={'points_world_m':pts.astype(np.float64),'points_world_m_f32':pts.copy(),'faces':faces.copy(),
         'quality_star_point_output_to_input_row':np.arange(len(pts),dtype=np.int64),
         'quality_star_face_output_to_input_row':np.arange(n,dtype=np.int64),'source_input_face_rows':np.arange(n,dtype=np.int64),
         'source_input_vertex_quotient':np.arange(len(pts),dtype=np.int64),'quality_star_patch_union_offsets':off.copy(),
         'face_ancestry_offsets':off.copy(),'source_collapse_operation_rows':np.full(n,-1,np.int64),
         'typed_parent_candidate_path':np.asarray(['parent.npz']),'typed_parent_candidate_sha256':np.asarray(['a'*64]),
         'endpoint_quotient_parent_path':np.asarray(['parent.npz']),'endpoint_quotient_parent_sha256':np.asarray(['a'*64])}
    for i,lane in enumerate(core.LANES):
        vals=np.arange(n,dtype=np.int64)+100*i;src['face_source_'+lane]=vals.copy();src['quality_star_patch_union_face_source_'+lane]=vals.copy()
    for i,lane in enumerate(core.ANCESTRY_LANES):src['face_ancestry_'+lane]=np.arange(n,dtype=np.int64)+1000*i
    return vol,src

def parity(role):
    volume=fx.fixture_volume();surface=fx.fixture_surface(volume);targets=('Skin','Fat') if role=='I' else ('Fat',)
    vols={vr:fx.fixture_volume() for vr in targets}
    full=core.propose_endpoint_quotient(role,surface,vols,(0,1),drop=0,keep=1)
    cache=b.LocalQualityCache(core,role,surface,vols,targets)
    local=cache.evaluate((0,1),0,1,(),50e-6,allow_local_tie=(role=='I'))
    assert local['combined_native_bad_before']==full['combined_native_bad_before']
    assert local['combined_native_bad_after']==full['combined_native_bad_after']
    assert abs(local['combined_log_deficit_before']-full['combined_log_deficit_before'])<1e-8
    assert abs(local['combined_log_deficit_after']-full['combined_log_deficit_after'])<1e-8
    assert set(local['surface']['changed_parent_rows'])==set(full['surface']['face_output_to_parent_rows'][full['surface']['changed_output_rows']])
    op={'operation_id':1,'role':role,'edge_rows':[0,1],'drop_source_row':0,'keep_source_row':1,
        'changed_face_base_rows':local['surface']['changed_parent_rows'].tolist(),'star_face_base_rows':local['surface']['star_rows'].tolist(),
        'affected_base_cells':{},'changed_base_cells':{},'volume_delta_det6_integer':{},'drop_volume_rows':{},'keep_volume_rows':{}}
    qsrc=np.arange(len(surface['points_world_m_f32']),dtype=np.int64);qsrc[0]=1
    final_s,_=b.build_source_output(surface,qsrc,[op],{'path':'parent.npz','sha256':'a'*64})
    full_s=core.endpoint_surface_arrays(surface,full['surface'],parent_path='parent.npz',parent_sha='a'*64,operation_id=1)
    assert np.array_equal(final_s['faces'],full_s['faces'])
    for lane in core.LANES:
        assert np.array_equal(final_s['face_source_'+lane],full_s['face_source_'+lane])
        assert np.array_equal(final_s['quality_star_patch_union_face_source_'+lane],full_s['quality_star_patch_union_face_source_'+lane])
    for vr in targets:
        r=local['volumes'][vr];fr=full['volumes'][vr]
        assert r['native_bad_after']-r['native_bad_before']==fr['native_bad_after']-fr['native_bad_before']
        assert abs((r['native_deficit_after']-r['native_deficit_before'])-(fr['native_deficit_after']-fr['native_deficit_before']))<1e-8
        assert r['exact_signed_volume_delta_det6_integer']==fr['exact_signed_volume_delta_det6_integer']
        op['affected_base_cells'][vr]=r['affected_input_cell_rows'].tolist();op['changed_base_cells'][vr]=r['changed_input_cell_rows'].tolist()
        op['volume_delta_det6_integer'][vr]=str(r['exact_signed_volume_delta_det6_integer'])
        op['drop_volume_rows'][vr]=int(r['drop_volume_row']);op['keep_volume_rows'][vr]=int(r['keep_volume_row'])
        qv=np.arange(len(vols[vr]['points_world_m']),dtype=np.int64);qv[int(r['drop_volume_row'])]=int(r['keep_volume_row'])
        mapped=qv[np.asarray(vols[vr]['tetrahedra'],np.int64)];ss=np.sort(mapped,axis=1);valid=~np.any(ss[:,1:]==ss[:,:-1],axis=1);cm=np.flatnonzero(valid)
        unions={}
        for basecell in r['changed_input_cell_rows']:unions[int(np.searchsorted(cm,basecell))]=set(map(int,r['affected_input_cell_rows']))
        local_v=b.build_volume_output(vols[vr],qv,cm,unions,[op],{'path':'parent-volume.npz','sha256':'b'*64},vr)
        full_v=core.endpoint_volume_arrays(vols[vr],fr,parent_path='parent-volume.npz',parent_sha='b'*64)
        assert np.array_equal(local_v['tetrahedra'],full_v['tetrahedra'])
        assert np.array_equal(local_v['material_indices'],full_v['material_indices'])
        assert np.array_equal(local_v['desired_boundary_faces'],full_v['desired_boundary_faces'])
        assert np.array_equal(local_v['boundary_faces'],full_v['boundary_faces'])
    return True

def paired_gate_controls():
    # Exercise the admission rule independently of geometric trial generation:
    # Skin makes strict progress while Fat is already clear and remains a tie.
    c=object.__new__(b.LocalQualityCache)
    c.role='I';c.targets=('Skin','Fat');c.sp=np.asarray(((0,0,0),(1e-5,0,0)),np.float32)
    c.current={'Skin':{'bad':1,'score':1.0},'Fat':{'bad':0,'score':0.0}}
    c._surface_trial=lambda *args:{'fixture':'paired'}
    outcomes={'Skin':{'native_bad_before':1,'native_bad_after':0,'native_deficit_before':1.0,
                      'native_deficit_after':0.0,'native_bad_by_material_before':{},'native_bad_by_material_after':{}},
              'Fat':{'native_bad_before':0,'native_bad_after':0,'native_deficit_before':0.0,
                     'native_deficit_after':0.0,'native_bad_by_material_before':{},'native_bad_by_material_after':{}}}
    c._volume_trial=lambda vr,*args:outcomes[vr]
    good=c.evaluate((0,1),0,1,(),50e-6,allow_local_tie=True)
    assert good['volumes']['Skin']['native_bad_before']==1 and good['volumes']['Skin']['native_bad_after']==0
    assert good['volumes']['Fat']['native_bad_before']==0 and good['volumes']['Fat']['native_bad_after']==0
    assert good['combined_native_bad_before']==1 and good['combined_native_bad_after']==0

    # A tied role cannot conceal a score regression; standalone D cannot tie.
    outcomes['Fat']['native_deficit_after']=1e-5
    try:c.evaluate((0,1),0,1,(),50e-6,allow_local_tie=True)
    except ValueError as e:assert 'paired local tie worsens' in str(e)
    else:raise AssertionError('paired local tie score regression passed')
    outcomes['Fat']['native_deficit_after']=0.0
    outcomes['Skin'].update(native_bad_before=0,native_bad_after=0,native_deficit_before=0.0,native_deficit_after=0.0)
    c.current={'Skin':{'bad':0,'score':0.0},'Fat':{'bad':0,'score':0.0}}
    try:c.evaluate((0,1),0,1,(),50e-6,allow_local_tie=True)
    except ValueError as e:assert 'no strict combined native progress' in str(e)
    else:raise AssertionError('nonprogressing paired tie passed')
    c.role='D';c.targets=('Fat',);c.current={'Fat':{'bad':0,'score':0.0}}
    try:c.evaluate((0,1),0,1,(),50e-6,allow_local_tie=False)
    except ValueError as e:assert 'unpaired local quality has no strict progress' in str(e)
    else:raise AssertionError('standalone D tie passed')
    return True

def local_failure_controls():
    volume=fx.fixture_volume();surface=fx.fixture_surface(volume)
    def rejected(v,needle):
        cache=b.LocalQualityCache(core,'D',surface,{'Fat':v},('Fat',))
        try:cache._volume_trial('Fat',0,1)
        except ValueError as e:
            assert needle in str(e),(needle,str(e))
            return
        raise AssertionError('negative volume quotient control passed: '+needle)
    # Two distinct positive input tetrahedra become the same cell after redirect.
    dup={k:np.array(v,copy=True) for k,v in volume.items()}
    dp=np.asarray(((0,0,0),(1e-7,0,0),(0.5e-7,1,0),(0.5e-7,0,1),(0.5e-7,-1,-1)),np.float32)
    dup['points_world_m']=dp.copy();dup['points_world_m_f32']=dp.copy()
    dup['tetrahedra']=core.M.orient(dup['points_world_m_f32'],np.asarray(((0,2,3,4),(1,2,3,4)),np.int64))
    dup['material_indices']=np.full(2,16,np.int32);dup['desired_boundary_faces']=core.boundary_faces(dup['tetrahedra']);dup['boundary_faces']=dup['desired_boundary_faces'].copy()
    rejected(dup,'duplicate tetrahedra')
    # Both rows map exactly, but the dropped source endpoint has no volume star.
    orphan={k:np.array(v,copy=True) for k,v in volume.items()}
    orphan['tetrahedra']=orphan['tetrahedra'][2:3].copy();orphan['material_indices']=orphan['material_indices'][2:3].copy()
    orphan['desired_boundary_faces']=core.boundary_faces(orphan['tetrahedra']);orphan['boundary_faces']=orphan['desired_boundary_faces'].copy()
    rejected(orphan,'orphan volume endpoint')
    # A valid source edge / target chain can still fail the 3D link condition.
    assert b.volume_edge_link_ok({'points_world_m_f32':np.asarray([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,1),(1,2,1),(2,2,1),(3,2,1)],np.float32),
        'tetrahedra':np.asarray([(0,1,2,3),(0,4,5,6),(1,4,7,8)],np.int64)},
        {'points_world_m_f32':np.asarray([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,1),(1,2,1),(2,2,1),(3,2,1)],np.float32),
         'faces':np.asarray([(0,1,2),(1,0,3)],np.int64)},0,1) is False
    return True

def run():
    assert parity('D') and parity('I')
    assert paired_gate_controls() and local_failure_controls()
    vol,src=combine();basev={k:np.array(v,copy=True) for k,v in vol.items()};bases={k:np.array(v,copy=True) for k,v in src.items()}
    cache=b.LocalQualityCache(core,'D',src,{'Fat':vol},('Fat',))
    qsrc=np.arange(len(src['points_world_m_f32']),dtype=np.int64);qvol={'Fat':np.arange(len(vol['points_world_m']),dtype=np.int64)}
    ops=[]
    for serial,(drop,keep) in enumerate(((0,1),(6,7)),1):
        got=cache.evaluate((drop,keep),drop,keep,(),50e-6);sres=got['surface'];vr=got['volumes']['Fat']
        affected=np.asarray(vr['affected_input_cell_rows'],np.int64);changed=np.asarray(vr['changed_input_cell_rows'],np.int64)
        qsrc[drop]=qsrc[keep];qvol['Fat'][int(vr['drop_volume_row'])]=qvol['Fat'][int(vr['keep_volume_row'])]
        ops.append({'operation_id':serial,'role':'D','edge_rows':[drop,keep],'drop_source_row':drop,'keep_source_row':keep,
            'changed_face_base_rows':np.asarray(sres['changed_parent_rows'],np.int64).tolist(),
            'star_face_base_rows':np.asarray(sres['star_rows'],np.int64).tolist(),
            'affected_base_cells':{'Fat':affected.tolist()},'changed_base_cells':{'Fat':changed.tolist()},
            'volume_delta_det6_integer':{'Fat':str(vr['exact_signed_volume_delta_det6_integer'])},
            'drop_volume_rows':{'Fat':int(vr['drop_volume_row'])},'keep_volume_rows':{'Fat':int(vr['keep_volume_row'])}})
        cache.commit(got)
    face_out,fmap=b.build_source_output(bases,qsrc,ops,{'path':'parent.npz','sha256':'a'*64})
    mt=qvol['Fat'][np.asarray(basev['tetrahedra'],np.int64)];ss=np.sort(mt,axis=1);valid=~np.any(ss[:,1:]==ss[:,:-1],axis=1)
    cm=np.flatnonzero(valid);inv=np.full(len(valid),-1,np.int64);inv[cm]=np.arange(len(cm))
    unions={}
    for op in ops:
        for r in op['changed_base_cells']['Fat']:
            unions.setdefault(int(inv[r]),set()).update(op['affected_base_cells']['Fat'])
    volume_out=b.build_volume_output(basev,qvol['Fat'],cm,unions,ops,{'path':'parent-volume.npz','sha256':'b'*64},'Fat')
    target_len=len(volume_out['desired_boundary_faces'])
    for k,dtype in (('desired_boundary_source_group',np.int8),('desired_boundary_source_rows',np.int64),
        ('desired_boundary_source_face_rows',np.int64),('proposed_current_desired_boundary_source_group',np.int8),
        ('proposed_current_desired_boundary_source_rows',np.int64),('proposed_current_desired_boundary_source_face_rows',np.int64)):
        volume_out[k]=np.zeros(target_len,dtype=dtype)
    volume_out['fat_current_target_rebind_wall_faces_volume_rows']=np.empty((0,3),np.int64)
    volume_out['fat_current_target_rebind_wall_face_points_world_m_f32']=np.empty((0,3,3),np.float32)
    source_check=b.verify_final_surface(bases,face_out,qsrc,fmap,ops,{'path':'parent.npz','sha256':'a'*64})
    volume_check=b.verify_final_volume(basev,volume_out,qvol['Fat'],cm,unions,ops,IdentityRebinder,'Fat',{'D':face_out},
                                       {'path':'parent-volume.npz','sha256':'b'*64})
    assert cache.current['Fat']['bad']==0
    assert source_check['full_typed_union_and_ancestry_replayed'] and volume_check['full_replacement_parent_union_replayed']
    assert b.volume_edge_link_ok({'points_world_m_f32':np.asarray([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,1),(1,2,1),(2,2,1),(3,2,1)],np.float32),
        'tetrahedra':np.asarray([(0,1,2,3),(0,4,5,6),(1,4,7,8)],np.int64)},
        {'points_world_m_f32':np.asarray([(0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,1,1),(2,1,1),(1,2,1),(2,2,1),(3,2,1)],np.float32),
         'faces':np.asarray([(0,1,2),(1,0,3)],np.int64)},0,1) is False
    print({'status':'PASS_BATCH007_LOCAL_CACHE_TWO_DISJOINT_OPERATIONS','D_full_core_parity':True,'I_full_core_parity':True,'baseline_quality_scans':1,
        'local_star_quality_updates':2,'paired_Skin_1_to_0_Fat_0_to_0_tie':True,'D_tie_and_paired_score_regression_rejected':True,
        'duplicate_orphan_link_pinch_rejected':True,'final_source_and_volume_replay':True,
        'native_bad_before_after':(volume_check['native_bad_before'],volume_check['native_bad_after'])})

if __name__=='__main__':run()
