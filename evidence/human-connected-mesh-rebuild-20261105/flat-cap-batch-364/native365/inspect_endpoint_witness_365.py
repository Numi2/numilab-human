from pathlib import Path
import sys,json,hashlib,numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');N=W/'flat-cap-native-365';R=W/'matched-ccd-baseline-335'
sys.path.insert(0,str(W))
from compare_remeshed_native_310 import capture,sha
report=json.loads((N/'endpoint-matched-field-diagnostic.json').read_text());w=report['identical_cells']['maximum_witness']
def inspect(z,row):
 t=z['tets'][row];r=z['rest'][t];u=z['u'][t];dm=(r[1:]-r[:1]).T
 f=np.eye(3)+(u[1:]-u[:1]).T@np.linalg.inv(dm)
 e=(f.T@f-np.eye(3))*.5;sv=np.linalg.svd(dm,compute_uv=False)
 vol=abs(float(np.linalg.det(dm)))/6
 esq=sum(float(np.dot(r[i]-r[j],r[i]-r[j])) for i in range(4) for j in range(i))
 q=12*(3*vol)**(2/3)/esq
 return dict(tetrahedron=row,node_rows=t.tolist(),object=next(k for k,v in z['objects'].items() if v==z['tobj'][row]),region=int(z['treg'][row]),rest_m=r.tolist(),accepted_displacement_m=u.tolist(),node_force_channels_N=z['forces'][t].tolist(),node_mass_kg=z['mass'][t].tolist(),node_tags=z['tag'][t].tolist(),volume_m3=vol,quality=q,rest_singular_values_m=sv.tolist(),rest_condition_number=float(sv[0]/sv[-1]),minimum_altitude_m=min(3*vol/(np.linalg.norm(np.cross(r[j]-r[i],r[k]-r[i]))/2) for i,j,k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]),F=f.tolist(),J=float(np.linalg.det(f)),green_strain=e.tolist(),green_strain_norm=float(np.linalg.norm(e)),maximum_node_displacement_m=float(np.linalg.norm(u,axis=1).max()))
a=capture(R/'install-motion.stdout.log.gz');ar=inspect(a,w['reference_tet']);del a
b=capture(N/'install-motion.stdout.log.gz');br=inspect(b,w['candidate_tet']);del b
assert np.array_equal(np.sort(np.asarray(ar['rest_m']),axis=0),np.sort(np.asarray(br['rest_m']),axis=0))
error=float(np.linalg.norm(np.asarray(br['green_strain'])-ar['green_strain']))
assert abs(error-w['absolute_green_strain_difference'])<1e-10
result=dict(reference=ar,candidate=br,absolute_green_strain_difference=error,solver_control_comparison=report['solver_control_comparison'],physical_fidelity_qualified=False,classification='Same retained reference cell; large endpoint strain difference is real in the emitted compensated displacement field. Old control settings differ, so this cannot isolate the mesh contribution.',owner=dict(path=__file__,sha256=sha(Path(__file__))))
(N/'largest-strain-witness.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:{x:v[x] for x in ['tetrahedron','object','region','volume_m3','quality','rest_condition_number','minimum_altitude_m','green_strain_norm','J','maximum_node_displacement_m']} for k,v in [('reference',ar),('candidate',br)]},indent=2),flush=True)
