import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import relocate_human_skin_fat_interiors as r


def star():
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[.125,.125,2**-16]],dtype=np.float32).astype(float)
 t=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
 return p,t


def test_bounded_lp_improves_complete_star_and_exact_volume():
 p,t=star();out=p.copy();out[4]=r.chebyshev_point(p,t,4,.5)
 assert r.mesh.measure(out,t)[1].min()>r.mesh.measure(p,t)[1].min()*100
 assert np.linalg.norm(out[4]-p[4])<=.5
 assert np.array_equal(out[:4],p[:4])
 assert r.exact_star_volume_preserved(p,out,t)
 assert r.spectral_condition(out,t).max()<r.spectral_condition(p,t).max()


def test_common_lattice_handles_new_coordinate_precision():
 p,t=star();out=p.copy();out[4]=[2**-23,.25,.25]
 assert r.exact_star_volume_preserved(p,out,t)
 out[3,2]=2
 assert not r.exact_star_volume_preserved(p,out,t)


@pytest.mark.parametrize("material,density",[(15,1000.),(16,900.)])
def test_uniform_mass_preserves_mass_and_first_moment(material,density):
 p,t=star();out=p.copy();out[4]=r.chebyshev_point(p,t,4,.5)
 old=density*r.mesh.measure(p,t)[0]/6
 new=r.uniform_star_mass(p,t,out,old,material)
 assert np.isclose(new.sum(),old.sum(),rtol=0,atol=1e-13)
 assert np.allclose(new/(r.mesh.measure(out,t)[0]/6),density,rtol=1e-13)
 assert np.allclose(np.einsum('i,ij->j',new,out[t].mean(1)),np.einsum('i,ij->j',old,p[t].mean(1)),rtol=0,atol=1e-12)


def test_nonuniform_density_rejected():
 p,t=star();old=1000*r.mesh.measure(p,t)[0]/6;old[0]*=.99
 with pytest.raises(ValueError,match='nonuniform'):r.uniform_star_mass(p,t,p,old,15)


@pytest.mark.parametrize('bad',[-1,0,float('nan')])
def test_bad_bound_rejected(bad):
 p,t=star()
 with pytest.raises(ValueError):r.chebyshev_point(p,t,4,bad)


def test_nonpositive_and_missing_node_rejected():
 p,t=star();t[0,[1,2]]=t[0,[2,1]]
 with pytest.raises(ValueError):r.chebyshev_point(p,t,4,.5)
 p,t=star()
 with pytest.raises(ValueError):r.chebyshev_point(p,t,9,.5)


def test_mass_csv_checks_rows_not_just_values(tmp_path):
 f=tmp_path/'mass.csv';f.write_text('child_row,mass_kg\n0,1\n1,2\n')
 assert np.array_equal(r.read_mass(f,2),[1,2])
 f.write_text('child_row,mass_kg\n1,1\n0,2\n')
 with pytest.raises(ValueError):r.read_mass(f,2)


def relocation_fixture():
 p,t=star();p=(p*.001).astype(np.float32).astype(float);out=p.copy();out[4]=r.chebyshev_point(p,t,4,.0005)
 labels=dict(material_index=np.full(4,15),geometric_donor_body=np.full(4,20),anatomical_region_id=np.full(4,2))
 source=dict(points_reference_world_m_f64=p,volume_tetrahedra=t,replacement_parent_offsets=np.arange(5),replacement_parent_cell_rows=np.arange(4),**labels)
 candidate=dict(source,points_reference_world_m_f64=out,replacement_parent_offsets=np.arange(5)*4,replacement_parent_cell_rows=np.tile(np.arange(4),4))
 mass=1000*r.mesh.measure(p,t)[0]/6;nm=r.uniform_star_mass(p,t,out,mass,15)
 op=dict(node=4,source_cell_rows=list(range(4)),old_world_m=p[4].tolist(),new_world_m=out[4].tolist(),labels=[15,20,2])
 return source,candidate,[op],mass,nm,[]


def test_independent_replay():
 args=relocation_fixture();result=r.verify_relocations(*args)
 assert result['nodes']==1 and result['all_typed_and_actual_boundaries_fixed']


@pytest.mark.parametrize('failure',['pinned','boundary','incomplete','mass','ancestry','label','undocumented','repeated'])
def test_independent_replay_rejects_invalid_candidates(failure):
 src,dst,ops,mass,nm,typed=relocation_fixture()
 if failure=='pinned':typed=[4]
 if failure=='boundary':ops[0]['node']=0
 if failure=='incomplete':ops[0]['source_cell_rows']=[0,1,2]
 if failure=='mass':nm[0]*=.99
 if failure=='ancestry':dst['replacement_parent_cell_rows'][0]=1
 if failure=='label':dst['material_index']=np.full(4,16)
 if failure=='undocumented':dst['points_reference_world_m_f64'][0,0]=np.float32(.00001)
 if failure=='repeated':ops=ops*2
 with pytest.raises(ValueError):r.verify_relocations(src,dst,ops,mass,nm,typed)


def test_mass_lookup_accepts_only_declared_coordinate_changes():
 from prepare_human_local_mass import relocation_query_points
 src,dst,ops,*_=relocation_fixture()
 query,moved=relocation_query_points(src['points_reference_world_m_f64'],dst['points_reference_world_m_f64'],ops)
 assert np.array_equal(moved,[4]) and np.array_equal(query,dst['points_reference_world_m_f64'])
 with pytest.raises(ValueError):relocation_query_points(src['points_reference_world_m_f64'],dst['points_reference_world_m_f64'],[])
 with pytest.raises(ValueError):relocation_query_points(src['points_reference_world_m_f64'],dst['points_reference_world_m_f64'],ops*2)
