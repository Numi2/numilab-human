#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, time, traceback
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
RUN=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/independent-replay-001'
INPUT=H/'joint-D414-I466-target-475-root/fat-target397/fat-current-typed-target-rebound.npz'
BASE=H/'combined-D414-I466-volume-469-root/fat-D414-I466-composed-volume-proposal.npz'
BASE_REPORT=H/'combined-D414-I466-volume-469-root/report.json'
OUTPUT=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/run001/fat-D475-surface-quotient-volume-proposal.npz'
OPS=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/run001/operation-cell-star-map.npz'
TRANSFER_REPORT=H/'d-shortedge-quotient-475-next-001-preflight-100um/volume-transfer-001/run001/report.json'
EXPECTED={'input':'c1903e60b03aeb3a53592c8ecab020abae59eb8829899dece614d4a946f6fb04','base':'d4e17216ba617cde361d3972b80fff003f426a874f17c449499727eabeabbf17','base_report':'da64291a3f2d2c863dd1692376d2ad3392950772799a328a44fa1b686e992bee','output':'703011aba8e8bbc5590dc1cef61140eed5abe7f2eb92727f416608d1b521067d','ops':'0e2d58b3599cf4e834df34e5401aa314c32f831546deadb28f5c584bd9d04aed','transfer_report':'99c3ed29e23fb1210ce3860f32979b8ecf2cf1904696af5aa7a3acc1a6eb76f7'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def need(c,m):
 if not c:raise ValueError(m)
def chain(tris):
 a=np.asarray(tris,np.int64).reshape((-1,3));good=(a[:,0]!=a[:,1])&(a[:,1]!=a[:,2])&(a[:,0]!=a[:,2]);a=a[good]
 if not len(a):return np.empty(0,np.int64),np.empty(0,np.int64)
 s=np.sort(a,axis=1);inv=((a[:,0]>a[:,1]).astype(np.int8)+(a[:,0]>a[:,2]).astype(np.int8)+(a[:,1]>a[:,2]).astype(np.int8));sign=np.where((inv&1)==0,1,-1).astype(np.int64);P=np.int64(1<<20);key=(s[:,0]*P+s[:,1])*P+s[:,2];order=np.argsort(key,kind='stable');ks=key[order];ss=sign[order];starts=np.r_[0,np.flatnonzero(ks[1:]!=ks[:-1])+1];sums=np.add.reduceat(ss,starts);keys=ks[starts];keep=sums!=0;return keys[keep],sums[keep]
def tet_boundary(t):
 t=np.asarray(t,np.int64);return np.concatenate((t[:,[1,2,3]],t[:,[0,3,2]],t[:,[0,1,3]],t[:,[0,2,1]]),axis=0)
def f32_units(points):
 bits=np.asarray(points,np.float32).view(np.uint32).reshape((-1,3));out=[]
 for row in bits:
  xyz=[]
  for raw in row:
   b=int(raw);sgn=-1 if b>>31 else 1;e=(b>>23)&255;frac=b&0x7fffff
   if e==255:raise ValueError('nonfinite Float32 point')
   if e==0:m=frac;shift=0
   else:m=(1<<23)|frac;shift=e-1
   xyz.append(sgn*(m<<shift))
  out.append(tuple(xyz))
 return out
def exact_det6(tets,units):
 vals=[]
 for tet in np.asarray(tets,np.int64):
  p0,p1,p2,p3=(units[int(i)] for i in tet);a=[p1[j]-p0[j] for j in range(3)];b=[p2[j]-p0[j] for j in range(3)];c=[p3[j]-p0[j] for j in range(3)];vals.append(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))
 return vals
def metrics(points,tets):
 t=np.asarray(tets,np.int64);p=np.asarray(points[t],np.float64);a=p[:,1]-p[:,0];b=p[:,2]-p[:,0];c=p[:,3]-p[:,0];det=np.einsum('ij,ij->i',a,np.cross(b,c));edges=[]
 for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):
  e=p[:,i]-p[:,j];edges.append(np.einsum('ij,ij->i',e,e))
 maxedge=np.maximum.reduce(edges);q=np.abs(det)/np.power(maxedge,1.5);vol=np.abs(det)/6.;ab=vol<=1e-18;nb=q<1e-5;return {'det6':det,'volume':vol,'q':q,'bad':ab|nb,'absolute_bad':ab,'normalized_bad':nb}
start=time.monotonic()
try:
 for name,path in [('input',INPUT),('base',BASE),('base_report',BASE_REPORT),('output',OUTPUT),('ops',OPS),('transfer_report',TRANSFER_REPORT)]:
  actual=sha(path);need(actual==EXPECTED[name],f'{name} hash mismatch: {actual}')
 inp=np.load(INPUT,allow_pickle=False);base=np.load(BASE,allow_pickle=False);out=np.load(OUTPUT,allow_pickle=False);op=np.load(OPS,allow_pickle=False);src=json.loads(BASE_REPORT.read_text());tr=json.loads(TRANSFER_REPORT.read_text())
 need(np.array_equal(inp['points_world_m_f32'],base['points_world_m_f32']),'input/base points differ');need(np.array_equal(inp['tetrahedra'],base['tetrahedra']),'input/base tets differ');need(np.array_equal(inp['material_indices'],base['material_indices']),'input/base material differs');need(int(src.get('nonpositive',-1))==0,'qualified base is not exact-positive')
 need(np.array_equal(inp['points_world_m_f32'],out['points_world_m_f32']),'output point bytes changed')
 node=np.asarray(out['input_node_to_output_node'],np.int64);need(node.shape==(len(inp['points_world_m_f32']),),'node map shape');need(np.all((node>=0)&(node<len(out['points_world_m_f32']))),'node map out of range')
 cellmap=np.asarray(out['input_cell_to_output_cell'],np.int64);ret=np.asarray(out['retained_input_cell_rows'],np.int64);out2in=np.asarray(out['output_cell_to_input_cell'],np.int64);need(cellmap.shape==(len(inp['tetrahedra']),),'cellmap shape');need(np.array_equal(ret,np.flatnonzero(cellmap>=0)),'retained rows != cell map');need(np.array_equal(out2in,ret),'output cell map differs');need(np.array_equal(cellmap[ret],np.arange(len(ret),dtype=np.int64)),'cell maps not inverse')
 rows=np.asarray(op['operation_changed_cell_rows'],np.int64);old=np.asarray(op['operation_old_changed_tetrahedra'],np.int64);new=np.asarray(op['operation_new_changed_tetrahedra'],np.int64);deleted=np.asarray(op['operation_changed_deleted'],bool);need(len(rows)==len(old)==len(new)==len(deleted),'operation ledger size');need(np.array_equal(inp['tetrahedra'][rows],old),'old op tets mismatch');need(np.array_equal(cellmap[rows]<0,deleted),'deleted flags mismatch');removed=np.flatnonzero(cellmap<0);need(np.array_equal(np.sort(removed),np.sort(np.asarray(op['operation_removed_cell_rows'],np.int64))),'removed rows mismatch');affected=np.unique(np.asarray(op['operation_full_star_cell_rows'],np.int64));need(np.array_equal(np.sort(affected),np.sort(rows)),'full-star != changed rows')
 keep=~deleted;out_tets=np.asarray(out['tetrahedra'],np.int64);newout=cellmap[rows[keep]];need(np.array_equal(out_tets[newout],new[keep]),'new operation tets mismatch output');expected=np.asarray(node[inp['tetrahedra'][ret]],np.int64);expected[newout]=new[keep];need(np.array_equal(expected,out_tets),'output tet table fails complete row replay');need(np.array_equal(inp['material_indices'][ret],out['material_indices']),'material rows changed')
 unitsin=f32_units(inp['points_world_m_f32']);unitsout=f32_units(out['points_world_m_f32']);olddet=exact_det6(old,unitsin);newdet=exact_det6(out_tets[newout],unitsout);need(all(x>0 for x in olddet),'affected source tets not positive');need(all(x>0 for x in newdet),'replacement tets not positive');delta=sum(newdet)-sum(olddet);need(str(delta)==str(tr['exact_F32_local_det6_delta_integer']),'exact det6 delta mismatch');den=1<<447;need(str(den)==str(tr['exact_F32_det6_scale_denominator']),'exact det6 scale mismatch')
 im=metrics(inp['points_world_m_f32'],inp['tetrahedra']);om=metrics(out['points_world_m_f32'],out_tets);om_old=metrics(inp['points_world_m_f32'],old);om_new=metrics(out['points_world_m_f32'],out_tets[newout]);regress=int(np.count_nonzero((~om_old['bad'])&om_new['bad']));improve=int(np.count_nonzero(om_old['bad']&(~om_new['bad'])))
 srcchain=chain(tet_boundary(inp['tetrahedra']));srcboundary=chain(inp['boundary_faces']);need(all(np.array_equal(a,b) for a,b in zip(srcchain,srcboundary)),'source tets/stored boundary disagree');mapped=chain(node[np.asarray(inp['boundary_faces'],np.int64)]);actual=chain(tet_boundary(out_tets));stored=chain(out['boundary_faces']);need(all(np.array_equal(a,b) for a,b in zip(actual,stored)),'output tets/stored boundary disagree');need(all(np.array_equal(a,b) for a,b in zip(mapped,actual)),'output boundary differs from mapped source chain')
 report={'schema':'h706.d475-independent-indexed379-replay.v1','status':'PASS_INDEPENDENT_D475_VOLUME_REPLAY','script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),'input_path':str(INPUT),'input_sha256':sha(INPUT),'qualified_positive_base_path':str(BASE),'qualified_positive_base_sha256':sha(BASE),'qualified_positive_base_report_sha256':sha(BASE_REPORT),'candidate_path':str(OUTPUT),'candidate_sha256':sha(OUTPUT),'operation_ledger_path':str(OPS),'operation_ledger_sha256':sha(OPS),'transfer_report_path':str(TRANSFER_REPORT),'transfer_report_sha256':sha(TRANSFER_REPORT),'input_cells':len(inp['tetrahedra']),'output_cells':len(out_tets),'input_points':len(inp['points_world_m_f32']),'output_points':len(out['points_world_m_f32']),'removed_input_cells':len(removed),'affected_cell_rows':len(rows),'affected_full_star_rows':len(affected),'affected_replacement_tets':len(newdet),'quotiented_input_point_ids':int(len(node)-len(np.unique(node))),'all_point_coordinate_bytes_preserved':True,'all_output_cells_reconstructed_from_input_and_operation_ledger':True,'material_rows_preserved':True,'all_input_exact_positive_inherited_from_pinned_base':True,'affected_old_exact_positive_count':len(olddet),'replacement_exact_positive_count':len(newdet),'exact_det6_delta_integer':str(delta),'exact_det6_scale_denominator':str(den),'exact_volume_delta_m3':float(delta/(6*den)),'input_native_bad_cells_recomputed':int(np.count_nonzero(im['bad'])),'output_native_bad_cells_recomputed':int(np.count_nonzero(om['bad'])),'input_absolute_volume_failures_recomputed':int(np.count_nonzero(im['absolute_bad'])),'output_absolute_volume_failures_recomputed':int(np.count_nonzero(om['absolute_bad'])),'input_normalized_quality_failures_recomputed':int(np.count_nonzero(im['normalized_bad'])),'output_normalized_quality_failures_recomputed':int(np.count_nonzero(om['normalized_bad'])),'changed_old_good_to_new_bad_count':regress,'changed_bad_to_good_count':improve,'input_oriented_boundary_face_count':len(inp['boundary_faces']),'output_oriented_boundary_face_count':len(out['boundary_faces']),'input_boundary_reconstructed_from_tets':True,'output_boundary_reconstructed_from_tets':True,'output_boundary_equals_mapped_input_oriented_chain':True,'transfer_target_residual_missing':int(tr['mapped_target_boundary_residual']['missing_oriented_records']),'transfer_target_residual_extra':int(tr['mapped_target_boundary_residual']['extra_oriented_records']),'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','limitations':['Candidate remains a volume quotient proposal with the transfer-reported residual target-boundary mismatch.','Independent replay verifies pinned input, operation ledger, output connectivity/material, changed-cell exact signs, and oriented boundary-chain mapping.','Exact positivity for unchanged cells is inherited from the exact-positive 469 base, whose geometry is byte-identical to replay input.','No source mass mapping, final O/I/D joint context, full target closure, or native execution is qualified.'],'elapsed_s':time.monotonic()-start}
 p=RUN/'report.json';p.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':report['status'],'receipt':str(p),'sha256':sha(p),'exact_det6_delta_integer':str(delta),'bad_before':report['input_native_bad_cells_recomputed'],'bad_after':report['output_native_bad_cells_recomputed'],'old_good_regressions':regress,'elapsed_s':round(report['elapsed_s'],3)},sort_keys=True))
except Exception as exc:
 RUN.mkdir(parents=True,exist_ok=True);r={'schema':'h706.d475-independent-indexed379-replay.v1','status':'FAIL_INDEPENDENT_D475_VOLUME_REPLAY','script_path':str(Path(__file__)),'script_sha256':sha(Path(__file__)),'error':repr(exc),'traceback':traceback.format_exc(),'elapsed_s':time.monotonic()-start};p=RUN/'failure.json';p.write_text(json.dumps(r,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':r['status'],'receipt':str(p),'sha256':sha(p),'error':r['error']},sort_keys=True));raise
