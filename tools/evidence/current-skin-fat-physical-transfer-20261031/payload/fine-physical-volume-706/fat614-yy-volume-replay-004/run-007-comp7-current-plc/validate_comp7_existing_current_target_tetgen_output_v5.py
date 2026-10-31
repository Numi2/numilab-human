#!/usr/bin/env python3
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
OUT=H/'fat614-yy-volume-replay-004/run-007-comp7-current-plc';CASE=H/'fat614-compact-cavities-004-remesh-002/component-0007';BASE=H/'fat614-residual-pass-001/remeshed/connected-fixed-surface-volume-proposal.npz';PREP=H/'fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz';PARENT=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz';D602=H/'fat614-minimal-f32-target-split-001/candidate-D-minimally-rounded-yy-edge-split.npz';DFINAL=H/'fat614-minimal-f32-target-split-002/run-005/candidate-D-remaining-yy-target-edge-splits.npz';DREPORT=H/'fat614-minimal-f32-target-split-002/run-005/report.json';DAUDIT=H/'fat614-minimal-f32-target-split-002/exact-audit-002/report.json';DLOADER=H/'fat614-minimal-f32-target-split-002/run-005/actual-697-adapter-001/receipt.json';BIN=H/'attribute-propagation-062/hull-attribute-fix-001/tetgen-hull-attribute-fix';OPTS='-pMYYAS0O0/0/0T9.9999999999999998e-17';CI=7
EXPECTED={'parent':'d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f','prep':'5810a3139b619630b8658c407a3b22cde1939ba9c917d273852622743ce5fe44','d602':'602036c9b164bd1eec83b344675b8cabf453665920e45b558b50fe8115c67493','dfinal':'4ce01a4bd397e7cf2b3c469ee33a3937d9173ff7d585a2687a1a09cf98377ab2','dreport':'3fef3f823f952afc67b9c481d9ce452414a8c4d7aa3c2420fca5f0f461da9e9c','daudit':'8eaf3426ff00f12691592c9e1f7ce923f6d011c2a0a5dcb009056d194e289854','dloader':'c8bc0ca2293d242672ce575622bcbf86d75bfdf82704c093678644e57599e858','old_poly':'82ad9a320393e182ec639898a31223417b529b5ad29dd34dd4e2cbee035e6936','tetgen':'01d15d76f8dfcdcd84f10fac280f7aeed3de1ae24991b0a359e05006c99e6bcb'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p):return json.JSONDecoder().raw_decode(Path(p).read_text())[0]
def fkey(x):return np.asarray(x,np.float32).view(np.uint32).tobytes()
def face_key_xyz(keys):return tuple(sorted(keys))
def oriented_key(keys):
 s=tuple(sorted(keys));inv=sum(keys[i]>keys[j] for i,j in ((0,1),(0,2),(1,2)));return (s,1 if inv%2==0 else -1)
def read_poly(path):
 raw=[x.rstrip('\n') for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')];i=0
 point_head=raw[i];ph=point_head.split();i+=1;n=int(ph[0]);point_rows=raw[i:i+n];i+=n
 point_map={}
 for line in point_rows:
  q=line.split();point_map[int(q[0])]=np.asarray([float(v) for v in q[1:4]],np.float32)
 facet_header=raw[i];fh=facet_header.split();i+=1;nf=int(fh[0]);facets=[]
 for fi in range(nf):
  h=raw[i].split();i+=1;npoly=int(h[0]);nhole=int(h[1]);marker=int(h[2]) if len(h)>2 else 0
  polygons=[]
  for _ in range(npoly):
   q=raw[i].split();i+=1
   if int(q[0])!=3:raise ValueError('non-triangle PLC facet unsupported')
   polygons.append([int(v) for v in q[1:4]])
  hole_rows=raw[i:i+nhole];i+=nhole
  if npoly!=1 or nhole:raise ValueError('expected one triangle and no facet holes')
  facets.append({'vertices':polygons[0],'marker':marker,'raw_header':raw[i-nhole-1-len(polygons)] if False else None})
 holes_head=raw[i];nholes=int(holes_head.split()[0]);i+=1;hole_rows=raw[i:i+nholes];i+=nholes
 regions_head=raw[i];nregions=int(regions_head.split()[0]);i+=1;region_rows=raw[i:i+nregions];i+=nregions
 if i!=len(raw):raise ValueError('unparsed .poly tail')
 return {'point_header':point_head,'points':point_map,'point_rows':point_rows,'facet_header':facet_header,'facets':facets,'holes_head':holes_head,'hole_rows':hole_rows,'regions_head':regions_head,'region_rows':region_rows}
def write_poly(path,data,points,facets):
 ph=data['point_header'].split();ph[0]=str(len(points));lines=['# Rebuilt from exact current D target; source-only component-7 boundary.',' '.join(ph)]
 for nid,x,marker in points:lines.append(f'{int(nid)} {float(np.float32(x[0])):.17g} {float(np.float32(x[1])):.17g} {float(np.float32(x[2])):.17g} {int(marker)}')
 fh=data['facet_header'].split();fh[0]=str(len(facets));lines.append(' '.join(fh))
 for face,marker in facets:lines.extend([f'1 0 {int(marker)}',f'3 {int(face[0])} {int(face[1])} {int(face[2])}'])
 lines.append(data['holes_head']);lines.extend(data['hole_rows']);lines.append(data['regions_head']);lines.extend(data['region_rows']);Path(path).write_text('\n'.join(lines)+'\n')
def is_closed_oriented(faces):
 ed=[]
 for f in np.asarray(faces,np.int64):ed.extend(((int(f[0]),int(f[1])),(int(f[1]),int(f[2])),(int(f[2]),int(f[0]))))
 und=defaultdict(list)
 for a,b in ed:und[tuple(sorted((a,b)))].append(1 if a<b else -1)
 return all(len(v)==2 and sum(v)==0 for v in und.values()),len(und),sum(len(v)!=2 or sum(v)!=0 for v in und.values())
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod);return mod
def load_npz(path):
 with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def table(path):
 lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')];n=int(lines[0].split()[0]);a=np.asarray([[float(v) for v in s.split()] for s in lines[1:1+n]],np.float64)
 if len(a)!=n:raise ValueError(f'{path} row count mismatch')
 return a
def exact_dets(p,t):
 used=np.unique(np.asarray(t,np.int64).ravel());lut=np.full(len(p),-1,np.int64);lut[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used]);return [int(m.exact_det(ip,[int(lut[int(x)]) for x in tet])) for tet in np.asarray(t,np.int64)]
def oriented_counter(faces):
 c=Counter()
 for f in np.asarray(faces,np.int64):
  a,b,d=map(int,f);k=tuple(sorted((a,b,d)));inv=int(a>b)+int(a>d)+int(b>d);c[(k,1 if inv%2==0 else -1)]+=1
 return +c
def exact_flux(p,b):
 used=np.unique(np.asarray(b,np.int64).ravel());lut=np.full(len(p),-1,np.int64);lut[used]=np.arange(len(used));ip=m.integer_points(np.asarray(p,np.float32)[used]);v=0
 for f in b:
  a,c,d=[ip[int(lut[int(x)])] for x in f];v+=a[0]*(c[1]*d[2]-c[2]*d[1])-a[1]*(c[0]*d[2]-c[2]*d[0])+a[2]*(c[0]*d[1]-c[1]*d[0])
 return int(v)
def prepare():
 for name,path,want in [('parent',PARENT,EXPECTED['parent']),('prep',PREP,EXPECTED['prep']),('d602',D602,EXPECTED['d602']),('dfinal',DFINAL,EXPECTED['dfinal']),('dreport',DREPORT,EXPECTED['dreport']),('daudit',DAUDIT,EXPECTED['daudit']),('dloader',DLOADER,EXPECTED['dloader']),('old_poly',CASE/'cavity.poly',EXPECTED['old_poly']),('tetgen',BIN,EXPECTED['tetgen'])]:
  got=sha(path)
  if got!=want:raise ValueError(f'{name} SHA mismatch {got}')
 report=read_json(DREPORT);audit=read_json(DAUDIT);loader=read_json(DLOADER)
 if audit.get('physical_admission') is not False or audit.get('native_steps')!=0 or loader.get('physical_admission') is not False:raise ValueError('D target source-only audit/loader gate')
 runops=[o for o in report['operations'] if int(o['component_id'])==CI]
 if len(runops)!=1:raise ValueError(f'expected one D target split op for comp7, got {len(runops)}')
 op=runops[0]
 with np.load(PREP,allow_pickle=False) as z:
  offsets=np.asarray(z['face_offsets'],np.int64);po=np.asarray(z['point_offsets'],np.int64);grows=np.asarray(z['global_point_rows'][int(po[CI]):int(po[CI+1])],np.int64);localfaces=np.asarray(z['local_faces'][int(offsets[CI]):int(offsets[CI+1])],np.int64)
 with np.load(PARENT,allow_pickle=False) as z:parentp=np.asarray(z['points_world_m'],np.float32)
 d602=load_npz(D602);dfinal=load_npz(DFINAL);d602p=np.asarray(d602['points_world_m_f32'],np.float32);dfp=np.asarray(dfinal['points_world_m_f32'],np.float32)
 if not np.array_equal(dfp[:len(d602p)].view(np.uint32),d602p.view(np.uint32)):raise ValueError('D point prefix mismatch')
 pkeys={fkey(x):i for i,x in enumerate(parentp)};global_points=[x.copy() for x in parentp]
 for x in dfp:
  k=fkey(x)
  if k not in pkeys:pkeys[k]=len(global_points);global_points.append(np.asarray(x,np.float32).copy())
 global_points=np.asarray(global_points,np.float32);drow_to_global={i:pkeys[fkey(x)] for i,x in enumerate(dfp)}
 if np.max(grows)>=len(parentp):raise ValueError('prepared volume point row exceeds parent point domain')
 shell=grows[localfaces].copy();old_shell=shell.copy();edge=tuple(map(int,op['edge_global_volume_point_rows']));split_global=int(drow_to_global[int(op['new_output_D_point_row'])]);
 if split_global<len(parentp):raise ValueError('declared D split point is not appended target point')
 d602_to_global={i:pkeys[fkey(x)] for i,x in enumerate(d602p)};parent_rows=sorted({int(p['parent_face_row']) for p in op['face_patches']});parent_global=[np.asarray([d602_to_global[int(v)] for v in d602['faces'][fr]],np.int64) for fr in parent_rows]
 if len(parent_global)!=2:raise ValueError('expected two D parent triangles')
 output=[];replaced=[]
 for f in shell:
  hit=[q for q in parent_global if set(map(int,f))==set(map(int,q))]
  if not hit:output.append(f.tolist());continue
  if len(hit)!=1:raise ValueError('ambiguous D parent triangle in component shell')
  replaced.append(f.tolist());u,v=edge;tri=list(map(int,f));found=None
  for k in range(3):
   a,b,w=tri[k],tri[(k+1)%3],tri[(k+2)%3]
   if {a,b}=={u,v}:found=(a,b,w);break
  if found is None:raise ValueError('split target edge not in parent shell triangle')
  a,b,w=found;output.extend([[a,split_global,w],[split_global,b,w]])
 expected_shell=np.asarray(output,np.int64)
 if len(replaced)!=2 or len(expected_shell)!=len(shell)+2:raise ValueError('component-7 shell does not have the expected 2-to-4 D split')
 ok,edges,edge_failures=is_closed_oriented(expected_shell)
 if not ok:raise ValueError(f'updated component shell is not closed/oriented: {edge_failures}')
 old=read_poly(CASE/'cavity.poly');poly_coord_to_id={fkey(x):int(n) for n,x in old['points'].items()}
 if len(poly_coord_to_id)!=len(old['points']):raise ValueError('duplicate input PLC coordinates')
 for row in grows:
  if fkey(parentp[int(row)]) not in poly_coord_to_id:raise ValueError(f'prepared boundary point {int(row)} missing from old PLC')
 split_key=fkey(global_points[split_global])
 if split_key in poly_coord_to_id:raise ValueError('declared D split point already present; stale-target assumption false')
 old_actual=Counter(tuple(sorted(fkey(old['points'][int(n)]) for n in f['vertices'])) for f in old['facets'])
 expected_keys=Counter(tuple(sorted(fkey(global_points[int(n)]) for n in f)) for f in expected_shell)
 missing=expected_keys-old_actual;extra=old_actual-expected_keys
 expected_parent_keys=Counter(tuple(sorted(fkey(parentp[int(n)]) for n in f)) for f in replaced)
 child_keys=Counter(tuple(sorted(fkey(global_points[int(n)]) for n in f)) for f in expected_shell if split_global in f)
 if sum(missing.values())!=4 or sum(extra.values())!=2 or extra!=expected_parent_keys or missing!=child_keys:raise ValueError(f'old PLC discrepancy is not exactly the D 2-to-4 split: missing={sum(missing.values())}, extra={sum(extra.values())}')
 # Bind split triangles to the audited output rows, ignoring only the known Fat/D orientation reversal.
 child_auth=[]
 for patch in op['face_patches']:
  for tri in patch['child_faces_D_rows']:child_auth.append(tuple(sorted(fkey(dfp[int(v)]) for v in tri)))
 expected_child_coords=[tuple(sorted(fkey(global_points[int(v)]) for v in f)) for f in expected_shell if split_global in f]
 if Counter(child_auth)!=Counter(expected_child_coords):raise ValueError('component split children differ from audited D output triangles')
 # Rebuild all triangle facets from the expected current shell, preserving markers on unchanged and inherited parent facets.
 marker_by_key=defaultdict(list)
 for f in old['facets']:
  k=tuple(sorted(fkey(old['points'][int(n)]) for n in f['vertices']));marker_by_key[k].append(int(f['marker']))
 node_points=[(int(n),x.copy(),0) for n,x in old['points'].items()];newid=max(old['points'])+1;node_points.append((newid,global_points[split_global].copy(),0));allnodes={fkey(x):n for n,x,_ in node_points}
 if len(allnodes)!=len(node_points):raise ValueError('rebuilt PLC point coordinate collision')
 facets=[];marker_hist=Counter()
 for f in expected_shell:
  keys=[fkey(global_points[int(v)]) for v in f]
  if any(k not in allnodes for k in keys):raise ValueError('current shell vertex missing from rebuilt PLC points')
  canon=tuple(sorted(keys));marks=marker_by_key.get(canon)
  if not marks:
   # New child inherits the single authenticated parent facet marker.
   parent=next((k for k in expected_parent_keys if set(k)<=set(canon) or len(set(k)&set(canon))==2),None)
   if parent is None:raise ValueError('cannot inherit a facet marker for new D child')
   mark=marker_by_key[parent][0]
  else:mark=marks[0]
  facets.append(([allnodes[k] for k in keys],mark));marker_hist[mark]+=1
 if len(facets)!=306 or len({tuple(sorted(f)) for f,_ in facets})!=len(facets):raise ValueError('rebuilt PLC facet count or uniqueness mismatch')
 poly_path=OUT/'cavity.poly';write_poly(poly_path,old,node_points,facets)
 # Read the just-written PLC back and compare full oriented face/coordinate identity.
 rebuilt=read_poly(poly_path)
 got=Counter(oriented_key([fkey(rebuilt['points'][int(n)]) for n in f['vertices']]) for f in rebuilt['facets'])
 want=Counter(oriented_key([fkey(global_points[int(n)]) for n in f]) for f in expected_shell)
 if got!=want:raise ValueError('rebuilt PLC oriented triangles differ from the audited current shell')
 # No authored point moves; exactly one declared D target point is appended.
 for n,x,_ in node_points[:-1]:
  if fkey(x)!=fkey(old['points'][n]):raise ValueError('old PLC point changed')
 prep_report={'schema':'numi.fat614.component-current-D-target-plc.v1','status':'PASS_REBUILT_COMP7_PLC_TO_AUDITED_D_TARGET','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','component_id':CI,'old_poly_path':str(CASE/'cavity.poly'),'old_poly_sha256':sha(CASE/'cavity.poly'),'expected_old_missing_facets':4,'expected_old_extra_facets':2,'old_extra_are_D_parent_faces':True,'missing_are_four_audited_D_children':True,'prepared_shell_faces':len(shell),'updated_target_shell_faces':len(expected_shell),'old_poly_point_count':len(old['points']),'rebuilt_poly_point_count':len(node_points),'declared_D_split_point_row':int(op['new_output_D_point_row']),'new_poly_node_id':newid,'new_D_point_f32':global_points[split_global].tolist(),'D_parent_face_rows':parent_rows,'D_child_face_rows':op['face_patches'],'rebuilt_facet_marker_histogram':dict(marker_hist),'closed_oriented_shell':True,'D_candidate_sha256':sha(DFINAL),'D_full_audit_sha256':sha(DAUDIT),'D_697_loader_sha256':sha(DLOADER),'parent_volume_sha256':sha(PARENT),'tetgen_binary_sha256':sha(BIN),'tetgen_options':OPTS,'poly_path':str(poly_path),'poly_sha256':sha(poly_path)}
 prep_path=OUT/'prepare.json';prep_path.write_text(json.dumps(prep_report,indent=2)+'\n');print(json.dumps(prep_report));return prep_path

def run_once():
 # Validation-only derivative: TetGen ran once under v4 and emitted these files,
 # but v4 failed before it recorded the subprocess return code. Never rerun here.
 global m,topo,caps
 prep_path=OUT/'prepare.json'
 if not prep_path.is_file():raise ValueError('prepare phase missing')
 prep=json.loads(prep_path.read_text())
 if prep.get('status')!='PASS_REBUILT_COMP7_PLC_TO_AUDITED_D_TARGET' or sha(OUT/'cavity.poly')!=prep['poly_sha256']:raise ValueError('prepared PLC receipt/hash mismatch')
 outputs=[OUT/f'cavity.1.{ext}' for ext in ('node','ele','face','edge')]
 if not all(p.is_file() for p in outputs) or (OUT/'run.json').exists():raise FileExistsError('validation-only path requires frozen TetGen outputs and no prior run receipt')
 r=load('fat614_comp7_validator_remesh296',H/'remesh_fat614_compact_cavities_296_007.py');topo=r.m;m=r.m.m
 caps=load('fat614_comp7_validator_caps288',H/'direct_connected_skin_caps_288_root.py')
 argv=[str(BIN),OPTS,'cavity.poly'];stdout=(OUT/'tetgen.stdout.log').read_text();stderr=(OUT/'tetgen.stderr.log').read_text()
 from types import SimpleNamespace
 proc=SimpleNamespace(returncode=None,stdout=stdout,stderr=stderr)
 result={'schema':'numi.fat614.comp7-current-target-yy-remesh.v2-validation-only','physical_admission':False,'native_steps':0,'mass_mapping_status':'NOT_RUN','prepare_receipt_path':str(prep_path),'prepare_receipt_sha256':sha(prep_path),'poly_sha256':sha(OUT/'cavity.poly'),'binary_sha256':sha(BIN),'argv_recorded_from_frozen_v4_source':argv,'return_code':None,'return_code_status':'NOT_CAPTURED_BEFORE_V4_VALIDATION_NAMEERROR','stdout_sha256':sha(OUT/'tetgen.stdout.log'),'stderr_sha256':sha(OUT/'tetgen.stderr.log'),'stdout_tail':stdout.splitlines()[-30:],'stderr_tail':stderr.splitlines()[-30:],'elapsed_s':None,'boundary_steiner_error':('Boundary contains Steiner points' in stdout+stderr),'tetgen_invocation_count_reported_by_v4_source':1,'validation_source':str(out),'validation_source_sha256':sha(out),'tetgen_outputs':{p.name:sha(p) for p in outputs}}
 if proc.returncode not in (0,None):
  result['status']='REJECT_TETGEN_CURRENT_TARGET_PLC';result['reason']='TetGen returned nonzero on the one bounded -YY invocation'
 else:
  # Full exact source-only validation of the emitted local tetrahedralization.
  node=table(OUT/'cavity.1.node');ele=table(OUT/'cavity.1.ele');face=table(OUT/'cavity.1.face');ids=node[:,0].astype(np.int64);raw=node[:,1:4].astype(np.float64);idslut={int(x):i for i,x in enumerate(ids)};tlocal=np.asarray([[idslut[int(x)] for x in rr] for rr in ele[:,1:5]],np.int64);rawp=raw.astype(np.float32)
  # Map input PLC coordinates to d4906 + all current target D points, then append emitted interior nodes.
  parent=load_npz(PARENT);pp=np.asarray(parent['points_world_m'],np.float32);tdesc=np.asarray(parent['tetrahedra'],np.int64);dfinal=load_npz(DFINAL);dfp=np.asarray(dfinal['points_world_m_f32'],np.float32);points=[x.copy() for x in pp];ix={fkey(x):i for i,x in enumerate(pp)}
  for x in dfp:
   k=fkey(x)
   if k not in ix:ix[k]=len(points);points.append(x.copy())
  newd=len(points)-len(pp);poly=read_poly(OUT/'cavity.poly');polyids=set(poly['points']);polykey={fkey(x):n for n,x in poly['points'].items()};usedids=set(map(int,ele[:,1:5].astype(np.int64).ravel()))|set(map(int,face[:,1:4].astype(np.int64).ravel()));ng=np.full(len(ids),-1,np.int64)
  for i,x in enumerate(rawp):
   if int(ids[i]) not in usedids:continue
   k=fkey(x)
   if k in ix:glob=ix[k]
   else:glob=len(points);ix[k]=glob;points.append(x.copy())
   ng[i]=glob
  if np.any(ng<0):raise ValueError('TetGen output used node map failed')
  newp=np.asarray(points,np.float32);gt=np.asarray([[int(ng[idslut[int(n)]]) for n in rr] for rr in ele[:,1:5]],np.int64)
  input_poly_missing=[]
  for nid,x in poly['points'].items():
   i=idslut.get(int(nid))
   if i is None or not np.array_equal(rawp[i].view(np.uint32),x.view(np.uint32)):input_poly_missing.append(int(nid))
  boundary_nodeids=set(map(int,face[:,1:4].astype(np.int64).ravel()));new_boundary=[int(ids[i]) for i in range(len(ids)) if int(ids[i]) not in polyids and int(ids[i]) in boundary_nodeids]
  det,q=m.measurements(newp,gt);ex=exact_dets(newp,gt);nativebad=int(np.sum(~m.passes(det,q)));nonpos=int(sum(x<=0 for x in ex))
  glfaces=np.asarray([[int(ng[idslut[int(n)]]) for n in rr] for rr in face[:,1:4].astype(np.int64)],np.int64);boundary=topo.boundary(gt)
  face_cn_ids=Counter(tuple(sorted(map(int,f))) for f in face[:,1:4].astype(np.int64))
  nodeid_to_global={int(ids[i]):int(ng[i]) for i in range(len(ids)) if ng[i]>=0}
  poly_global_faces=np.asarray([[nodeid_to_global[int(n)] for n in f['vertices']] for f in read_poly(OUT/'cavity.poly')['facets']],np.int64)
  global_to_nodeid={int(ng[i]):int(ids[i]) for i in range(len(ids)) if ng[i]>=0}
  boundary_node_faces=np.asarray([[global_to_nodeid[int(v)] for v in face] for face in boundary],np.int64)
  boundary_cn_ids=Counter(tuple(sorted(map(int,f))) for f in boundary_node_faces)
  local_source_shell=[]
  with np.load(PREP,allow_pickle=False) as z:
   po=z['point_offsets'];fo=z['face_offsets'];grows=z['global_point_rows'][int(po[CI]):int(po[CI+1])];lf=z['local_faces'][int(fo[CI]):int(fo[CI+1])];source_shell=grows[lf]
  oldprep=read_poly(CASE/'cavity.poly');old_mark=defaultdict(list)
  for f in oldprep['facets']:old_mark[tuple(sorted(fkey(oldprep['points'][int(n)]) for n in f['vertices']))].append(int(f['marker']))
  # Expected shell triangles are the exact target rebuilt in prepare.json, recovered from the emitted PLC file.
  plpoly=read_poly(OUT/'cavity.poly');poly_coords={int(n):x for n,x in plpoly['points'].items()};expect_keys=Counter(tuple(sorted(map(int,f))) for f,_ in [(x,y) for x,y in ((rec['vertices'],rec['marker']) for rec in plpoly['facets'])])
  if input_poly_missing or new_boundary:raise ValueError(f'YY boundary-point gate failed: moved/missing inputs={input_poly_missing}, new_boundary_nodes={new_boundary}')
  if ele.shape[1]<6 or np.any(ele[:,5].astype(np.int64)!=16):raise ValueError('TetGen material region attribute changed')
  if nonpos:raise ValueError(f'new comp7 mesh has {nonpos} exact F32 nonpositive cells')
  if face_cn_ids!=boundary_cn_ids:raise ValueError('TetGen .1.face differs from tetrahedral boundary')
  # The output boundary coordinates must equal the rebuilt .poly triangles (orientation as an unoriented facet set).
  expected_poly_cn=Counter(tuple(sorted(int(v) for v in f['vertices'])) for f in plpoly['facets'])
  if face_cn_ids!=expected_poly_cn:raise ValueError('emitted surface differs from rebuilt current target PLC')
  if oriented_counter(poly_global_faces)!=oriented_counter(boundary):raise ValueError('oriented TetGen boundary differs from rebuilt current target PLC')
  # Exact flux and volume on global point rows.
  local_vol=sum(ex);local_flux=exact_flux(newp,boundary)
  if local_vol!=local_flux:raise ValueError('exact local TetGen volume/flux mismatch')
  # Exact self-SAT and retained d4906 tets; exclude all seven pending source-cavity rows.
  local_tree=caps.TetBVH(newp,gt);local_sat=caps.check_intersections(newp,gt,gt,np.arange(len(gt),dtype=np.int64),[],local_tree,max_pairs=5_000_000)
  if not local_sat.get('pass'):raise ValueError('comp7 local self SAT failed: '+str(local_sat))
  part=load_npz(H/'fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz');prep=load_npz(PREP);pmap={int(v):i for i,v in enumerate(part['retained_input_cell_rows'])};dmap={int(v):i for i,v in enumerate(parent['retained_input_cell_rows'])};co=prep['cell_offsets'];selected=[]
  for ci in (3,4,5,7,8,10,11):
   src=np.asarray(prep['input_cell_rows'][int(co[ci]):int(co[ci+1])],np.int64);pr=np.asarray([pmap[int(v)] for v in src],np.int64);selected.extend(int(dmap[int(v)]) for v in pr)
  tree=caps.TetBVH(pp,tdesc);external_sat=caps.check_intersections(newp,tdesc,gt,np.asarray(selected,np.int64),[],tree,max_pairs=10_000_000)
  if not external_sat.get('pass'):raise ValueError('comp7 retained-volume exact SAT failed: '+str(external_sat))
  usedpts=np.unique(gt);candidate=OUT/'component-0007-current-target-volume-proposal.npz';np.savez_compressed(candidate,points_world_m=newp[usedpts],tetrahedra=np.searchsorted(usedpts,gt),output_point_global_volume_rows=usedpts,material_indices=np.full(len(gt),16,np.int32),tetgen_output_node_ids=ids,output_node_global_volume_rows=ng)
  result.update({'status':'PASS_COMPONENT7_CURRENT_TARGET_YY_LOCAL_VOLUME','local_tets':len(gt),'points':len(usedpts),'exact_F32_positive_cells':len(gt),'exact_F32_nonpositive_cells':0,'native_quality_failures':nativebad,'strict_native_quality_thresholds_unchanged':True,'input_poly_points_preserved_bitwise':True,'new_boundary_nodes':0,'material_attribute16_preserved':True,'tetgen_face_matches_PLC':True,'exact_local_volume_equals_flux':True,'local_exact_SAT':local_sat,'retained_d4906_exact_SAT':external_sat,'component_candidate_path':str(candidate),'component_candidate_sha256':sha(candidate),'native_admission':'NOT_RUN','mass_mapping':'NOT_RUN'})
 path=OUT/'run.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
 if a.prepare_only==a.run:raise SystemExit('choose exactly one of --prepare-only or --run')
 if a.prepare_only:prepare()
 else:run_once()
if __name__=='__main__':main()
