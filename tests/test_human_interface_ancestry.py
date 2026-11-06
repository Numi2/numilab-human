from pathlib import Path
import sys,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_human_interface_ancestry as m

def fixture():
 p=np.array([[-1,0,0],[0,1,2**-16],[1,0,0],[0,-1,2**-16]],np.float32)
 old=np.array([[0,1,2],[0,2,3]]);new=np.array([[0,1,3],[1,2,3]])
 parent={'points_world_m_f32':p,'faces':old};child={'points_world_m_f32':p,'faces':new}
 for offset,lanes in [('face_ancestry_offsets',['face_ancestry_source_face_rows','face_ancestry_source_original_ids','face_ancestry_parent_face_rows']),('quality_star_patch_union_offsets',['quality_star_patch_union_face_source_'+k for k in ['layer','body','bone','face_id','original_id','identity0','identity1']])]:
  parent[offset]=np.array([0,1,2]);child[offset]=np.array([0,2,4])
  for k in lanes:parent[k]=np.array([7,8]);child[k]=np.array([7,8,7,8])
 return child,parent,np.arange(4),np.arange(2)

def test_exact_boundary_and_both_complete_ancestries_replay():
 r=m.verify_interface_replay(*fixture());assert r=={'patches':1,'maximum_gap_m':2**-16}

def test_forged_ancestry_is_rejected():
 c,p,q,r=fixture();c['quality_star_patch_union_face_source_body'][0]=99
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r)

def test_flipped_face_and_excessive_gap_are_rejected():
 c,p,q,r=fixture();c['faces'][0]=c['faces'][0,::-1]
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r)
 c,p,q,r=fixture()
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r,maximum_gap_m=1e-6)

def test_point_quotient_and_unpaired_change_are_rejected():
 c,p,q,r=fixture();q[0]=1
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r)
 c,p,q,r=fixture();c['faces'][1]=p['faces'][1]
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r)


def test_invalid_csr_cannot_claim_union_provenance():
 c,p,q,r=fixture();c['face_ancestry_offsets']=np.array([0,3,2])
 with pytest.raises(ValueError):m.verify_interface_replay(c,p,q,r)
