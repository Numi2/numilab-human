"""Generate static two-tetra geometry fixtures; no physical time stepping."""
from pathlib import Path
import struct,json,subprocess
root=Path('/Users/n/human-cardiac-wall-anatomy-20260912/checker-selftests')
root.mkdir(exist_ok=True)
exe=Path('/Users/n/human-cardiac-wall-anatomy-20260912/cardiac-wall-asset-check')
def packed(code,values):return struct.pack('<'+str(len(values))+code,*values)
points=[0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,0.,1.,0.,0.,-1.]
cells=[0,1,2,3,0,2,1,4]
faces=[[1,2,3],[0,3,2],[0,1,3],[2,1,4],[0,4,1],[0,2,4]]
flat=[i for face in faces for i in face]
base={'nodes.f64le':packed('d',points),'tetrahedra.u32le':packed('I',cells),'labels.u32le':packed('I',[1,2]),'source_reversed_cells.u32le':b'','fibres.f64le':packed('d',[1.,0.,0.]*2),'sheets.f64le':packed('d',[0.,1.,0.]*2),'boundary.u32le':packed('I',flat),'boundary_owners.u32le':packed('I',[0,0,0,1,1,1]),'boundary_components.u32le':packed('I',[0]*6)}
base.update({f'uvc_{name}.f64le':packed('d',[0.]*5) for name in ('rho','phi','z','v')})
changed_face=flat.copy();changed_face[1],changed_face[2]=changed_face[2],changed_face[1]
interior_face=flat.copy();interior_face[:3]=[0,2,1]
invalid_index=cells.copy();invalid_index[0]=5
negative=cells.copy();negative[0],negative[1]=negative[1],negative[0]
cases={
'valid_two_region':{},
'one_byte_truncated_coordinate':{'nodes.f64le':base['nodes.f64le'][:-1]},
'one_index_truncated_tet':{'tetrahedra.u32le':base['tetrahedra.u32le'][:-4]},
'out_of_range_cell_index':{'tetrahedra.u32le':packed('I',invalid_index)},
'inverted_cell':{'tetrahedra.u32le':packed('I',negative)},
'wrong_label_length':{'labels.u32le':packed('I',[1])},
'unknown_label':{'labels.u32le':packed('I',[1,25])},
'missing_fibre_buffer':{'fibres.f64le':None},
'nonfinite_fibre':{'fibres.f64le':packed('d',[float('nan'),0.,0.,1.,0.,0.])},
'nonfinite_uvc':{'uvc_rho.f64le':packed('d',[float('inf'),0.,0.,0.,0.])},
'short_sheet_buffer':{'sheets.f64le':base['sheets.f64le'][:-8]},
'reversed_boundary':{'boundary.u32le':packed('I',changed_face)},
'interior_as_boundary':{'boundary.u32le':packed('I',interior_face)},
'wrong_owner':{'boundary_owners.u32le':packed('I',[0,0,0,0,1,1])},
'wrong_component':{'boundary_components.u32le':packed('I',[0,0,0,1,0,0])},
'duplicate_reversal_provenance':{'source_reversed_cells.u32le':packed('I',[0,0])},
}
results=[]
for name,patch in cases.items():
 directory=root/name;directory.mkdir(exist_ok=True)
 for file,data in (base|patch).items():
  p=directory/file
  if data is None:
   if p.exists():p.unlink()
  else:p.write_bytes(data)
 run=subprocess.run([str(exe),str(directory)],capture_output=True,text=True)
 parsed=json.loads(run.stdout)
 if name=='valid_two_region':
  assert run.returncode==0 and parsed['status']=='pass',parsed
  assert abs(parsed['total_tetrahedron_volume_m3']-1/3)<1e-15
  assert parsed['material_interface_faces']==1 and parsed['boundary_faces']==6 and len(parsed['regions'])==2
  assert all(abs(region['volume_m3']-1/6)<1e-15 for region in parsed['regions'])
 else:assert run.returncode!=0 and parsed['status']=='failed',(name,run.returncode,parsed)
 results.append({'case':name,'returncode':run.returncode,'status':parsed['status'],'error':parsed.get('error')})
report={'status':'pass','cases':len(results),'rejected_malformed_cases':len(results)-1,'physical_steps':0,'results':results}
p=Path('/Users/n/human-cardiac-wall-anatomy-20260912/evidence/cpp-malformed-tests.json');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
