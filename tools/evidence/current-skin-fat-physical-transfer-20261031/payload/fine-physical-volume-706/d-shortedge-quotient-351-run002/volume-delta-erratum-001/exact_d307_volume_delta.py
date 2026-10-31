#!/usr/bin/env python3
"""Exact dyadic local volume/boundary delta for the frozen D307 quotient."""
from pathlib import Path
import hashlib,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=H/'d-shortedge-quotient-351-run002';BASE=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz';OUT=R/'volume-run003/fat-D-shortedge-quotient-volume-proposal.npz';TRANSFER=R/'volume-run003/report.json';STARS=R/'volume-run003/affected-fat-stars-001.npz';REPLAY=R/'independent-replay-002/report.json';OUTDIR=R/'volume-delta-erratum-001'
PINS={str(BASE):'d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f',str(OUT):'93435e624d47f7cfbf253df709871c6b4d170e9ea6b0b9c5867104b1b6f2875d',str(TRANSFER):'9d64de9f5b316ff2583b473fb50a6e772f1dd7d4271b118918b83462d9a52328',str(STARS):'0f62550ea189dcb0012092cfb544ab9f4bb72bcfd43f0157e01d6304a8743322',str(REPLAY):'b42ea68bee49a629842c9f9a9d6b69c435a77be52463691e9c1cd2fd26eea975'}
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def lattice(points):
 out=[]
 for row in np.asarray(points,np.float32):
  rr=[]
  for x in row:
   n,d=float(x).as_integer_ratio();rr.append(int(n)<<(149-(d.bit_length()-1)))
  out.append(tuple(rr))
 return out
def det4(ip,t):
 a,b,c,d=[ip[int(i)] for i in t];u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];w=[d[i]-a[i] for i in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
def inv3(f):
 return (f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)
def boundary_sig(t):
 f=np.asarray(t,np.int64)[:,PAT].reshape(-1,3);s=np.sort(f,axis=1);sign=1-2*(inv3(f)%2);o=np.lexsort((s[:,2],s[:,1],s[:,0]));s=s[o];sign=sign[o]
 starts=np.r_[0,1+np.flatnonzero(np.any(s[1:]!=s[:-1],axis=1))];cnt=np.diff(np.r_[starts,len(s)]);ss=np.add.reduceat(sign,starts)
 if np.any(cnt>2) or np.any(ss[cnt==2]!=0):raise ValueError('material boundary chain invalid')
 return np.c_[s[starts[cnt==1]],ss[cnt==1]]
def map_sig(sig,node_inverse):
 tri=node_inverse[np.asarray(sig[:,:3],np.int64)];coeff=np.asarray(sig[:,3],np.int64)*(1-2*(inv3(tri)%2));sort=np.sort(tri,axis=1);order=np.lexsort((sort[:,2],sort[:,1],sort[:,0]));return np.c_[sort[order],coeff[order]]
def flux_integer(ip,sig):
 total=0
 for row in sig:
  a,b,c=[ip[int(i)] for i in row[:3]];cross=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0]);total+=int(row[3])*(a[0]*cross[0]+a[1]*cross[1]+a[2]*cross[2])
 return total
def run():
 t0=time.monotonic();OUTDIR.mkdir(parents=True,exist_ok=False)
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError('pinned input changed '+p)
 b=load(BASE);o=load(OUT);tr=json.loads(TRANSFER.read_text());rr=json.loads(REPLAY.read_text());
 if rr['status']!='PASS_INDEPENDENT_D307_VOLUME_MAP_BOUNDARY_QUALITY_MATERIAL_REPLAY':raise ValueError('independent replay does not pass')
 p0=b['points_world_m'];t0a=b['tetrahedra'];m0=b['material_indices'];p1=o['points_world_m'];t1=o['tetrahedra'];m1=o['material_indices'];nodeinv=o['output_node_to_input_node'];cellmap=o['input_cell_to_output_cell']
 with np.load(STARS,allow_pickle=False) as z:affected=np.asarray(z['affected_volume_cell_rows'],np.int64)
 if len(affected)!=2286:raise ValueError('affected-star count')
 exactp=lattice(p0);materials=sorted(set(map(int,m0))|set(map(int,m1)));results=[]
 # Exact tet-volume delta uses only cells in the complete affected-star union; all remaining tets are identical by replay.
 affected_set=set(map(int,affected))
 for mat in materials:
  rows=np.asarray([r for r in affected if int(m0[int(r)])==mat],np.int64)
  dsum=0
  for r in rows:
   before=det4(exactp,t0a[int(r)]);outrow=int(cellmap[int(r)])
   after=0 if outrow<0 else det4(exactp,nodeinv[t1[outrow]])
   dsum+=after-before
  ib=np.flatnonzero(m0==mat);ob=np.flatnonzero(m1==mat)
  in_sig=boundary_sig(t0a[ib]);out_sig=map_sig(boundary_sig(t1[ob]),nodeinv)
  # Explicitly enforce the mapped material boundary chain equals the original chain after applying the node quotient.
  # The operation map is the final input->output node map; duplicate faces collapse out of the chain.
  with np.load(OUT,allow_pickle=False) as zo: nodemap=np.asarray(zo['input_node_to_output_node'],np.int64)
  mapped_faces=nodemap[np.asarray(in_sig[:,:3],np.int64)]
  sd=np.sort(mapped_faces,axis=1);keep=~np.any(sd[:,1:]==sd[:,:-1],axis=1)
  mapped_chain=map_sig(in_sig[keep],np.full(len(p1),-1,dtype=np.int64)) if False else None
  # Rebuild the expected source material boundary in output row space, preserving each source face's orientation.
  tri=nodemap[in_sig[:,:3]];deg=np.any(np.sort(tri,axis=1)[:,1:]==np.sort(tri,axis=1)[:,:-1],axis=1);tri=tri[~deg];coef=in_sig[~deg,3]*(1-2*(inv3(tri)%2));ss=np.sort(tri,axis=1);ordr=np.lexsort((ss[:,2],ss[:,1],ss[:,0]));expected=np.c_[ss[ordr],coef[ordr]]
  if not np.array_equal(expected,boundary_sig(t1[ob])):raise ValueError(f'material {mat} boundary differs under exact quotient')
  in_flux=flux_integer(exactp,in_sig);out_flux=flux_integer(exactp,out_sig);bdiff=out_flux-in_flux
  if dsum!=bdiff:raise ValueError(f'material {mat} exact tet-volume delta != exact oriented boundary flux delta: {dsum} vs {bdiff}')
  # Lattice coordinate is 2^-149 m, so signed volume is det_sum / (6*2^447).
  delta=float(dsum)/(6.0*(2.0**447))
  declared=tr['volume']['total_physical_volume_by_material_m3'][str(mat)]
  results.append({'material':mat,'affected_parent_cells':len(rows),'exact_delta_determinant_sum_on_2^-149_lattice':str(dsum),'lattice_scale_exponent':149,'exact_volume_delta_m3':delta,'exact_boundary_flux_delta_determinant_sum':str(bdiff),'boundary_equals_tet_delta_exact_integer':True,'producer_float_delta_m3':float(declared['delta_m3']),'float_delta_difference_m3':delta-float(declared['delta_m3']),'note':'Producer total-volume delta is preserved as emitted; exact local integer delta is the cancellation-free authoritative geometric difference.'})
 report={'schema':'h706.d307-exact-local-volume-delta-erratum.v1','status':'PASS_EXACT_LOCAL_TET_AND_ORIENTED_BOUNDARY_VOLUME_DELTA','physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN','source_pins':{p:sha(p) for p in PINS},'prior_independent_replay':{'path':str(REPLAY),'sha256':sha(REPLAY)},'method':{'domain':'complete affected drop+keep input cell-star union; all outside cells are unchanged by the pinned quotient replay','coordinate_lattice':'Float32 exact values represented as integers in units of 2^-149 m','tet_volume_units':'sum of signed 4x4 determinant integer differences divided by 6*2^447','boundary_units':'sum of oriented face triple-product integer differences divided by 6*2^447'},'results':results,'preservation':'No producer file or float delta field was modified; this is a versioned correction/precision receipt for the frozen candidate.','elapsed_s':time.monotonic()-t0}
 out=OUTDIR/'report.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'report_path':str(out),'report_sha256':sha(out),'results':results,'elapsed_s':report['elapsed_s']},indent=2))
if __name__=='__main__':run()
