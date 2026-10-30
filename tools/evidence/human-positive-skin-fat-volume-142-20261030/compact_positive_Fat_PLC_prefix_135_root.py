from pathlib import Path
import numpy as np,json,hashlib,time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');out=H/'fat-positive-PLC-prefix-compaction-135-root';out.mkdir(exist_ok=False);t0=time.monotonic()
source=H/'fat-inferred-boundary-positive-volume-127-root/fat-positive-inferred-boundary-proposal.npz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source)=='23ceea4acb0ab21783ac2bc7856e7389bf232b3a12be6842ef707150c9b8ce07'
with np.load(source)as z:p=z['points_world_m'];t=z['tetrahedra'];mat=z['material_indices']
used=np.zeros(len(p),bool);used[t.reshape(-1)]=True;original_plc_count=245996;assert used[:original_plc_count].all()
promoted=np.array([246200,246207,246228,246229,246234,246266,246270,246277],np.int64);assert used[promoted].all()
other=np.flatnonzero(used& (np.arange(len(p))>=original_plc_count)& ~np.isin(np.arange(len(p)),promoted))
order=np.r_[np.arange(original_plc_count,dtype=np.int64),promoted,other];inverse=np.full(len(p),-1,np.int64);inverse[order]=np.arange(len(order));nt=inverse[t]
assert len(order)==len(np.unique(order)) and np.all(nt>=0);newp=p[order]
for off in range(0,len(t),100000):assert np.array_equal(newp[nt[off:off+100000]].view(np.uint32),p[t[off:off+100000]].view(np.uint32))
assert np.array_equal(newp[:original_plc_count].view(np.uint32),p[:original_plc_count].view(np.uint32))
cert=H/'fat-inferred-boundary-exact-certification-129-v4/certification.json';cj=json.loads(cert.read_text())
outfile=out/'fat-positive-compact-PLC-prefix.npz'
np.savez_compressed(outfile,points_world_m=newp,tetrahedra=nt,material_indices=mat,
 source_candidate_sha256=np.array([sha(source)]),point_output_to_input_rows=order,point_input_to_output_row=inverse,
 cell_output_to_input_rows=np.arange(len(t),dtype=np.int64),promoted_inferred_D_boundary_input_node_rows=promoted,
 promoted_inferred_D_boundary_output_node_rows=np.arange(original_plc_count,original_plc_count+len(promoted),dtype=np.int64),
 original_PLC_point_count=np.array([original_plc_count]),new_PLC_point_count=np.array([original_plc_count+len(promoted)]))
with np.load(outfile)as z:
 for k,v in [('points_world_m',newp),('tetrahedra',nt),('material_indices',mat)]:assert np.array_equal(z[k],v)
report={'status':'PASS_BITWISE_GEOMETRY_PRESERVING_NODE_COMPACTION','candidate_path':str(outfile),'candidate_sha256':sha(outfile),'source_candidate_path':str(source),'source_candidate_sha256':sha(source),'source_independent_exact_certificate_path':str(cert),'source_independent_exact_certificate_sha256':sha(cert),'points_before':len(p),'points_after':len(newp),'unused_construction_points_removed':np.flatnonzero(~used).tolist(),'cells':len(t),'source_cell_order_unchanged':True,'every_cell_vertex_Float32_bits_unchanged':True,'material_region_attributes_unchanged':True,'inferred_D_promoted_input_node_rows':promoted.tolist(),'expected_new_Fat_PLC_prefix_points':original_plc_count+len(promoted),'actual004_assembly_prefix_verification':'PENDING_REFINED_D_ADAPTER','inherited_exact_sign_counts':{'positive':len(t),'negative':0,'zero':0},'mass_mapping':'NOT_RUN','physical_admission':False,'runtime':0,'elapsed_s':time.monotonic()-t0}
(out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':report['status'],'points_after':len(newp),'removed':int((~used).sum()),'cells':len(t),'candidate_sha256':report['candidate_sha256'],'elapsed_s':report['elapsed_s']}),flush=True)
