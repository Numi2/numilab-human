
"""Canonicalize exact composed Fat target metadata and immediate cell-parent unions."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
s=importlib.util.spec_from_file_location('canon540',H/'transfer_surface_contractions_indexed_379_root.py');T=importlib.util.module_from_spec(s);s.loader.exec_module(T)
out=H/'fat-canonical-current-target-540-root';out.mkdir(exist_ok=False);start=time.monotonic()
b=H/'fat-comp0-connected-warning-repair-412/run-044-current-54381-500um-preflight'
rp=b/'final-joint-354-D500-I528-O530-001/fat-target397/report.json';r=json.loads(rp.read_text());e=r['output_candidate'];cp=Path(e['path']);assert sha(cp)==e['sha256']=='ca0324b031fa57b0012ffc79c13a9bdaf4de92aa987a14c194f2f1280c1162e6'
ap=b/'final-joint-354-D500-I528-O530-001/audit354/report.json';ar=json.loads(ap.read_text());assert ar['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185' and ar['changed_vs_complete_retained_OID_both_walls_all185']['count']==ar['changed_OID_joint_self_pairs']['count']==0
cfgp=Path(ar['config']['path']);assert sha(cfgp)==ar['config']['sha256'];cfg=json.loads(cfgp.read_text());sources=cfg['candidate_surfaces']
for role,x in sources.items():assert sha(x['path'])==x['sha256']
a=load(cp);p=a['points_world_m'];t=a['tetrahedra'];mat=a['material_indices'];oldpath=Path(str(a['D500_I528_comp0_parent_path'][0]));assert sha(oldpath)==str(a['D500_I528_comp0_parent_sha256'][0])=='54138c6a3cbc62ff37a0bf9d86257896587304a493f9e05f9c9ec02fc3dc80fb';old=load(oldpath)
assert np.array_equal(p.view('u4'),old['points_world_m'].view('u4')) and np.array_equal(p.view('u4'),a['points_world_m_f32'].view('u4'))
origin=a['output_cell_to_input_cell'].copy();qmap=a['D500_I528_comp0_input_node_to_composed_node'];mapped=qmap[old['tetrahedra']];degen=np.any(np.diff(np.sort(mapped,axis=1),axis=1)==0,axis=1)
assert np.array_equal(origin,np.flatnonzero(~degen)) and np.array_equal(t,mapped[origin])
assert np.array_equal(mat,old['material_indices'][origin])
base_to_old=a['D500_I528_base_54381_input_cell_to_comp0_cell'];owner=np.full(len(old['tetrahedra']),-1,np.int64);groups=[]
def group(rows):
 rows=np.asarray(rows,np.int64);assert len(rows)>0 and len(rows)==len(np.unique(rows)) and np.all(rows>=0)
 assert np.all(owner[rows]<0);assert len(np.unique(old['material_indices'][rows]))==1
 owner[rows]=len(groups);groups.append(rows)
d=load(Path(str(a['D500_transfer_path'][0])));off=d['operation_cell_star_offsets'];rows=d['operation_cell_star_input_rows']
assert len(off)==1278
for k in range(len(off)-1):group(base_to_old[rows[off[k]:off[k+1]]])
ir=H/'paired-I-current-protected-528-root/exchange520/report.json';rr=json.loads(ir.read_text());ri=next(v for v in rr['roles'] if v['role']=='Fat');assert len(ri['operations'])==267
for o in ri['operations']:group(base_to_old[np.asarray(o['source_cell_rows'],np.int64)])
changed=np.any(t!=old['tetrahedra'][origin],axis=1);assert np.all(owner[origin[changed]]>=0)
mapping=origin.copy();mapping[changed]=-1;newrows=np.flatnonzero(changed);po=[0];pr=[]
for row in newrows:
 parent=groups[int(owner[origin[row]])];pr.extend(map(int,parent));po.append(len(pr))
assert np.array_equal(t[~changed],old['tetrahedra'][mapping[~changed]])
bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(a['desired_boundary_faces'])) and np.array_equal(T.boundary_identity(bd),T.boundary_identity(a['proposed_current_desired_boundary_faces']))
det,qual=T.m.measurements(p,t);assert np.all(det>0) and int((~T.m.passes(det,qual)).sum())==14366
changes={}
def replace(k,v):
 changes[k]={'before_shape':list(a[k].shape) if k in a else None,'before_sha256':hashlib.sha256(a[k].tobytes()).hexdigest() if k in a else None}
 a[k]=v
replace('output_cell_to_input_cell',mapping);reverse=np.full(len(old['tetrahedra']),-1,np.int64);reverse[mapping[~changed]]=np.flatnonzero(~changed);replace('input_cell_to_output_cell',reverse);replace('retained_input_cell_rows',mapping[~changed])
for k,v in {'current_geometry_parent_path':np.asarray([str(oldpath)]),'current_geometry_parent_sha256':np.asarray([sha(oldpath)]),'replacement_parent_offsets':np.asarray(po,np.int64),'replacement_parent_cell_rows':np.asarray(pr,np.int64),'replacement_tet_offsets':np.asarray([0,len(newrows)],np.int64),'replacement_tetrahedra':t[newrows]}.items():replace(k,v)
for role,x in sources.items():
 replace('fat_current_target_rebind_typed_'+role+'_candidate_path',np.asarray([x['path']]))
 replace('fat_current_target_rebind_typed_'+role+'_candidate_sha256',np.asarray([x['sha256']]))
 # Same explicit typed-source fields as other canonical physical layers.
 replace('typed_'+role+'_candidate_path',np.asarray([x['path']]))
 replace('typed_'+role+'_candidate_sha256',np.asarray([x['sha256']]))
dr=r['inputs']['typed_D_697_receipt'];assert sha(dr['path'])==dr['sha256'];ldr=json.loads(Path(dr['path']).read_text());assert ldr['status']=='PASS_ACTUAL_697_217_TYPED_D_LOAD' and ldr['typed_source_sha256']==sources['D']['sha256']
replace('D500_I528_typed_D_697_binding_status',np.asarray(['PASS_SOURCE_INGESTION_ONLY']))
replace('fat_current_target_rebind_metadata_mode',np.asarray(['Current O530/I528/D500 context; exact registered maps, target and immediate parent unions normalized; physical admission remains false']))
replace('fat_current_target_rebind_typed_D_697_receipt_path',np.asarray([dr['path']]))
replace('fat_current_target_rebind_typed_D_697_receipt_sha256',np.asarray([dr['sha256']]))
result=out/'fat-current-D500-I528-O530-canonical-volume-proposal.npz';np.savez_compressed(result,**a);written=load(result);before=load(cp)
assert set(written)==set(a) and all(np.array_equal(written[k],v) for k,v in a.items())
assert all(np.array_equal(written[k],v) for k,v in before.items() if k not in changes)
receipt={'status':'PASS_CANONICAL_CURRENT_FAT_TARGET_METADATA_AND_PARENT_UNIONS','physical_admission':False,'native_steps':0,'input':pin(cp),'input_target_report':pin(rp),'candidate':str(result),'candidate_sha256':sha(result),'geometry_parent':pin(oldpath),'source_surfaces':sources,'context':pin(ap),'D697':dr,'nodes':len(p),'cells':len(t),'native_quality_failures':14366,'nonpositive':0,'boundary_faces':len(bd),'missing_target_faces':0,'extra_actual_faces':0,'all_geometry_material_and_registered_maps_bit_identical':True,'all_point_coordinate_views_synchronized':True,'new_cell_parent_unions':len(newrows),'parent_rows':len(pr),'current_quotient_operation_groups':len(groups),'changed_metadata_fields':changes,'all_other_arrays_byte_identical':True,'source330_mass':'IMMUTABLE_COMPLETE_CURRENT_REBIND_REQUIRED','producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='changed_metadata_fields'}),flush=True)
