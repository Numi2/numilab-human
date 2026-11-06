from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_human_native_mesh_trials import strain_pair_witnesses

def fixture():
 p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
 return dict(rest=p,u=np.zeros_like(p),tets=np.array([[0,1,2,3]]),treg=np.array([7]))

def test_identical_field_is_exact_and_deterministic():
 a=fixture();r=strain_pair_witnesses(a,a)
 assert r['changed_rest_nodes']==0 and r['maximum_absolute_strain_difference_on_unchanged_cells']==0
 assert r['witnesses'][0]['tetrahedron']==0 and r['physical_admission'] is False

def test_known_affine_strain_and_source_change_classification():
 a=fixture();b=fixture();b['u'][:,0]=.1*b['rest'][:,0]
 r=strain_pair_witnesses(a,b);assert r['witnesses'][0]['absolute_green_strain_norm_difference']==pytest.approx(.105)
 assert r['witnesses'][0]['region']==7 and r['witnesses'][0]['rest_coordinates_unchanged']
 b['rest'][1,0]=1.1;r=strain_pair_witnesses(a,b)
 assert r['changed_rest_nodes']==1 and r['maximum_absolute_strain_difference_on_unchanged_cells']==0

@pytest.mark.parametrize('failure',['topology','nonfinite','degenerate','inverted','limit'])
def test_invalid_witness_input_rejected(failure):
 a=fixture();b=fixture();limit=12
 if failure=='topology':b['tets'][0,[1,2]]=b['tets'][0,[2,1]]
 if failure=='nonfinite':b['u'][0,0]=np.nan
 if failure=='degenerate':b['rest'][3]=b['rest'][0]
 if failure=='inverted':b['u'][3,2]=-2
 if failure=='limit':limit=0
 with pytest.raises(ValueError):strain_pair_witnesses(a,b,limit)


def test_exact_source_alignment_handles_node_and_cell_permutations():
 from compare_human_native_mesh_trials import align_source_indexed_mesh_pair
 a=fixture();a['tets']=a['tets'].astype(np.int32);a.update(objects={'skin':0},obj=np.zeros(4,int))
 order=np.array([2,0,3,1]);inverse=np.argsort(order)
 b=dict(rest=a['rest'][order],u=a['u'][order],tets=inverse[a['tets']][:,[2,1,0,3]],treg=a['treg'],objects=a['objects'],obj=a['obj'])
 raw=dict(points_reference_world_m_f64=a['rest'],volume_tetrahedra=a['tets'])
 aligned,receipt=align_source_indexed_mesh_pair(a,b,raw,raw,'skin')
 assert np.array_equal(aligned['rest'],a['rest']) and np.array_equal(aligned['tets'],a['tets'])
 assert receipt['permuted_nodes']==4 and receipt['nearest_node_matching'] is False
 assert strain_pair_witnesses(a,aligned)['maximum_accepted_displacement_difference_m']==0


def test_exact_alignment_rejects_unmapped_coordinates_and_labels():
 from compare_human_native_mesh_trials import align_source_indexed_mesh_pair
 a=fixture();a['tets']=a['tets'].astype(np.int32);a.update(objects={'skin':0},obj=np.zeros(4,int))
 raw=dict(points_reference_world_m_f64=a['rest'],volume_tetrahedra=a['tets'])
 b=dict(a,rest=a['rest'].copy());b['rest'][0,0]=.001
 with pytest.raises(ValueError):align_source_indexed_mesh_pair(a,b,raw,raw,'skin')
 b=dict(a,treg=np.array([8]))
 with pytest.raises(ValueError,match='anatomical'):align_source_indexed_mesh_pair(a,b,raw,raw,'skin')
 raw2=dict(raw,points_reference_world_m_f64=np.zeros((4,3)))
 with pytest.raises(ValueError,match='ambiguous'):align_source_indexed_mesh_pair(a,a,raw2,raw2,'skin')


def test_label_diagnostic_reports_without_admitting_or_relabeling():
 from compare_human_native_mesh_trials import align_source_indexed_mesh_pair
 a=fixture();a.update(objects={'skin':0},obj=np.zeros(4,int));b=dict(a,treg=np.array([8]))
 raw=dict(points_reference_world_m_f64=a['rest'],volume_tetrahedra=a['tets'])
 aligned,report=align_source_indexed_mesh_pair(a,b,raw,raw,'skin',report_label_differences=True)
 assert report['anatomical_label_differences']==1 and report['physical_admission'] is False
 assert report['anatomical_label_witnesses'][0]['candidate_region']==8 and aligned['treg'][0]==8
