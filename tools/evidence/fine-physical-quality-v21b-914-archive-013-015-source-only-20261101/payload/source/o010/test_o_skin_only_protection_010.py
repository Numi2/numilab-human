import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

core=load('o_skin_protection_core010',
    HERE.parent/'quality-o-endpoint-quotient-007-nativepreflight'/'outer_endpoint_quotient_007.py')
helper=load('o_skin_protection_helper010',HERE/'o_skin_only_protection_010.py')
owner855=helper.imp('pinned_protection_scope855_for_test010',helper.PROTECTION,helper.PROTECTION_SHA)
assert callable(owner855.point_key_index)
assert not hasattr(owner855.P,'point_key_index')


class FakeOwner:
    P=owner855.P
    point_key_index=staticmethod(owner855.point_key_index)
    def __init__(self,protection): self.protection=protection; self.requested=None
    def protected_scope(self,state_path,state,role,volumes,out):
        self.requested=(role,tuple(sorted(volumes)))
        assert role=='O'
        assert tuple(sorted(volumes))==('Skin',)
        return self.protection

def protection_doc(include_fat):
    rec={'back_rows':list(range(177)),'rim_rows':[177,178,179],
         'route_rows':[180,181]}
    result={'Skin':{'report':{'sources':{'O':rec}}}}
    if include_fat:
        # Deliberately unrelated Fat scope resembles its historical 790-row
        # union; O extraction must never read or constrain this record.
        result['Fat']={'report':{'sources':{
            'I':{'back_rows':list(range(790))},
            'actual_fat_back_union':{'current_rows':list(range(120)),
                                     'historical_rows':list(range(120,790))}}}}
    return result

def lineage_fixture():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="o010-lineage-") as td:
        td=Path(td)
        parent_points=np.asarray(((0,0,0),(1,0,0),(0,1,0),(0,0,1)),np.float32)
        child_points=parent_points[[0,2,3]].copy()
        quotient=np.asarray((0,0,1,2),np.int64)
        parent_path=td/"parent.npz"
        np.savez(parent_path,points_world_m_f32=parent_points)
        parent_ref={"path":str(parent_path),"sha256":helper.sha(parent_path)}
        child_path=td/"child.npz"
        np.savez(child_path,points_world_m_f32=child_points,
                 source_input_vertex_quotient=quotient,
                 typed_parent_candidate_path=np.asarray(str(parent_path)),
                 typed_parent_candidate_sha256=np.asarray(parent_ref["sha256"]))
        child_ref={"path":str(child_path),"sha256":helper.sha(child_path)}
        with np.load(child_path,allow_pickle=False) as z:child={k:np.array(z[k],copy=True) for k in z.files}
        with np.load(parent_path,allow_pickle=False) as z:parent={k:np.array(z[k],copy=True) for k in z.files}
        rows=np.arange(4,dtype=np.int64)
        mapped,proof,motion,immediate,final=owner855.lineage_rows_with_cumulative_l1(
            child_ref,parent_ref,child,parent,rows,"synthetic historical I/D",allow_collapsed=True)
        assert np.array_equal(mapped,quotient)
        assert proof["lineage_collapsed_rows"]==1 and proof["protected_reference_rows"]==4
        assert np.max(motion)>0 and np.array_equal(immediate,motion)
        assert np.array_equal(final,motion)
        try:
            owner855.lineage_rows_with_cumulative_l1(
                child_ref,parent_ref,child,parent,rows,"synthetic O back rows",allow_collapsed=False)
        except ValueError as exc:
            assert "collapsed" in str(exc)
        else:
            raise AssertionError("many-to-one quotient was accepted for O")
        # O's 177 hard-locked back rows remain unique and exactly fixed when the
        # current source is unchanged.
        o_points=np.arange(177*3,dtype=np.float32).reshape(177,3)
        o_path=td/"o-identity.npz"
        np.savez(o_path,points_world_m_f32=o_points)
        o_ref={"path":str(o_path),"sha256":helper.sha(o_path)}
        od={"points_world_m_f32":o_points}
        om,op,omotion,_,ofinal=owner855.lineage_rows_with_cumulative_l1(
            o_ref,o_ref,od,od,np.arange(177,dtype=np.int64),"synthetic O back rows",allow_collapsed=False)
        assert np.array_equal(om,np.arange(177)) and op["lineage_collapsed_rows"]==0
        assert not np.any(omotion) and not np.any(ofinal)
        return {"historical_I_D_records":4,"unique_current_representatives":3,
                "many_to_one_I_D_coverage":True,"same_quotient_rejected_for_O":True,
                "O_177_rows_unique_and_bitwise_fixed":True}

def run():
    lineage=lineage_fixture()
    source=np.arange(600,dtype=np.float32).reshape(200,3)
    skin=np.vstack((source,np.asarray([[900,901,902]],np.float32)))
    owner_lut,owner_ambiguous=owner855.point_key_index(skin)
    assert len(owner_lut)==len(skin) and not owner_ambiguous
    base=protection_doc(False); with_fat=protection_doc(True)
    got0=helper.o_rows_from_protection(base,len(source))
    got1=helper.o_rows_from_protection(with_fat,len(source))
    assert np.array_equal(got0[0],got1[0])
    assert np.array_equal(got0[1],got1[1])
    assert np.array_equal(got0[2],got1[2])
    assert got0[3]==got1[3]
    assert got0[3]=={'back_rows':177,'rim_rows':3,'route_rows':2,'source_rows':182}
    empty_optional={'Skin':{'report':{'sources':{'O':{
        'back_rows':list(range(177)),'rim_rows':[],'route_rows':[]}}}}}
    assert helper.o_rows_from_protection(empty_optional,len(source))[3]=={
        'back_rows':177,'rim_rows':0,'route_rows':0,'source_rows':177}
    mapped=core._exact_rows_to_volume(source,skin,got0[2])
    assert mapped=={int(r):int(r) for r in got0[2]}
    # The production wrapper gives the authenticated owner only the Skin
    # volume cache, even when both role volumes were supplied by the caller.
    owner=FakeOwner(base)
    surface={'points_world_m_f32':source}
    volume={'points_world_m_f32':skin}
    fixed,back,prot,p,detail=helper.protected_rows_with_owner(
        Path('/nonexistent/input-state.json'),{},'O',{'O':surface},
        {'Skin':volume,'Fat':{'points_world_m_f32':np.zeros((1,3),np.float32)}},
        HERE/'fixture-protection-output-010',core,owner)
    assert owner.requested==('O',('Skin',))
    assert len(back)==177 and len(fixed)==5
    assert set(map(int,detail['source_rows']))==set(map(int,got0[2]))
    assert np.array_equal(detail['volume_rows'],detail['source_rows'])
    assert set(prot)=={'Skin'}
    for bad in (
        {},
        {'Skin':{'report':{'sources':{}}}},
        {'Skin':{'report':{'sources':{'O':{'back_rows':[0],'rim_rows':[],'route_rows':[]}}}}},
        {'Skin':{'report':{'sources':{'O':{'back_rows':list(range(176))+[0],
                                           'rim_rows':[],'route_rows':[]}}}}},
    ):
        try: helper.o_rows_from_protection(bad,len(source))
        except ValueError: pass
        else: raise AssertionError('invalid O protection must fail closed')
    print({'status':'PASS_O_SKIN_ONLY_PROTECTION_010_CONTROLS',
           'skin_scope_identical_with_or_without_fat':True,
           'noncollapsed_exact_row_mapping':True,
           'owner_called_skin_only':True,'actual_855_lineage_owner_point_index_api':True,
           '177_O_back_rows_required':True,
           'malformed_or_duplicate_rows_rejected':True,
           'lineage_fixture':lineage})

if __name__=='__main__':run()
