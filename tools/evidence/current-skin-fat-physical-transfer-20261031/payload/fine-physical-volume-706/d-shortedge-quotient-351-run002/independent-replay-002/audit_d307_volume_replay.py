#!/usr/bin/env python3
"""Independent row-map, oriented-boundary, quality, material-volume and exact-F32 replay of D307 Fat quotient."""
from pathlib import Path
import hashlib, json, math, time
import numpy as np

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
R=H/'d-shortedge-quotient-351-run002'
BASE=H/'fat614-yy-volume-replay-002/fat-comp2-comp6-rounded-boundary-points-volume-proposal.npz'
BASE_REPORT=H/'fat614-yy-volume-replay-002/report.json'
OUT=R/'volume-run003/fat-D-shortedge-quotient-volume-proposal.npz'
VREPORT=R/'volume-run003/report.json'
SURFACE=R/'identity-seed-D.npz'
SREPORT=R/'surface-run001/report.json'
LEDGER=R/'surface-run001/accepted-operations.jsonl'
STARS=R/'volume-run003/affected-fat-stars-001.npz'
STARS_REPORT=R/'volume-run003/affected-fat-stars-001.json'
OUTDIR=R/'independent-replay-002'
EXPECTED={
 str(BASE):'d4906b176f78283f4073f9bd1e8d202a7301b57550a7b1b71fb101e76c4d236f',
 str(OUT):'93435e624d47f7cfbf253df709871c6b4d170e9ea6b0b9c5867104b1b6f2875d',
 str(VREPORT):'9d64de9f5b316ff2583b473fb50a6e772f1dd7d4271b118918b83462d9a52328',
 str(SURFACE):'7cab78eefc85112eda3c9f13df9839ecaadc58d82ac00f20b828d11d0ca8b535',
 str(SREPORT):'b549dd67254f0e44ae69c28f69545c5847235ca12dbdf69b98f7d3295d77156b',
 str(STARS):'0f62550ea189dcb0012092cfb544ab9f4bb72bcfd43f0157e01d6304a8743322',
 str(STARS_REPORT):'c4adfd86854da380c1bfcf123666f1e2e4a0b583b11db5f006ef175f4d2bfa5b',
}
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),dtype=np.int64)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def f32_key(p):
    return np.ascontiguousarray(np.asarray(p,dtype=np.float32)).view(np.uint32).tobytes()

def measurements(points,tets):
    q=np.asarray(points,np.float32)[np.asarray(tets,np.int64)].astype(np.float64)
    e=q[:,1:]-q[:,:1]
    det=np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))
    maxedge=np.maximum.reduce([np.linalg.norm(q[:,i]-q[:,j],axis=1) for i in range(4) for j in range(i+1,4)])
    nd=det/np.maximum(maxedge**3,np.finfo(np.float64).tiny)
    good=np.isfinite(det)&np.isfinite(nd)&(det/6.0>1e-18)&(nd>=1e-5)
    return det,nd,good

def face_signature(faces):
    f=np.asarray(faces,np.int64)
    if f.size==0:return np.empty((0,4),np.int64)
    s=np.sort(f,axis=1)
    inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)
    sign=1-2*(inv%2)
    order=np.lexsort((s[:,2],s[:,1],s[:,0]))
    return np.c_[s[order],sign[order]]

def oriented_boundary(tets):
    f=np.asarray(tets,np.int64)[:,PAT].reshape(-1,3)
    if len(f)==0:return np.empty((0,4),np.int64)
    s=np.sort(f,axis=1)
    inv=(f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8)
    sign=1-2*(inv%2)
    order=np.lexsort((s[:,2],s[:,1],s[:,0]))
    ss=s[order];sg=sign[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))]
    counts=np.diff(np.r_[starts,len(ss)])
    sums=np.add.reduceat(sg,starts)
    if np.any(counts>2):raise ValueError('tetrahedra have a nonmanifold face')
    if np.any(sums[counts==2]!=0):raise ValueError('internal face orientation does not cancel')
    if np.any(np.abs(sums[counts==1])!=1):raise ValueError('invalid oriented boundary coefficient')
    return np.c_[ss[starts[counts==1]],sums[counts==1]]

def boundary_flux(points,signature):
    if len(signature)==0:return 0.0
    f=np.asarray(signature[:,:3],np.int64).copy()
    neg=signature[:,3]<0
    f[neg,1],f[neg,2]=f[neg,2].copy(),f[neg,1].copy()
    xyz=np.asarray(points,np.float32).astype(np.float64)
    a=xyz[f[:,0]];b=xyz[f[:,1]];c=xyz[f[:,2]]
    return float(np.einsum('ij,ij->i',a,np.cross(b,c)).sum(dtype=np.float64)/6.0)

def f32_lattice_coords(points):
    # Exact integer representation on the common 2^-149 Float32 lattice.
    result=[]
    for row in np.asarray(points,np.float32):
        out=[]
        for x in row:
            n,d=float(x).as_integer_ratio()
            e=d.bit_length()-1
            out.append(int(n) << (149-e))
        result.append(tuple(out))
    return result

def exact_det(ip,t):
    a,b,c,d=[ip[int(i)] for i in t]
    u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)];w=[d[i]-a[i] for i in range(3)]
    return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def run():
    started=time.monotonic();OUTDIR.mkdir(parents=True,exist_ok=False)
    for p,h in EXPECTED.items():
        if sha(p)!=h:raise ValueError(f'frozen input SHA mismatch: {p}')
    base_report=json.loads(BASE_REPORT.read_text())
    transfer=json.loads(VREPORT.read_text())
    surface_report=json.loads(SREPORT.read_text())
    stars_report=json.loads(STARS_REPORT.read_text())
    if transfer['status']!='PASS_D_ONLY_POSITIVE_VOLUME_QUOTIENT_PARTIAL_BOUNDARY':raise ValueError('transfer report status')
    if transfer['volume']['input_sha256']!=EXPECTED[str(BASE)] or transfer['volume']['output_sha256']!=EXPECTED[str(OUT)]:raise ValueError('transfer report candidate pins')
    if surface_report['status']!='PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL':raise ValueError('surface report status')
    if surface_report['joint_changed_vs_complete_OID_walls_all185']['pair_count']!=0 or surface_report['joint_changed_vs_complete_OID_walls_all185']['new_face_self_pairs']!=0:raise ValueError('surface exact context gate not passing')
    if stars_report['status']!='PASS_EXACT_BASE_DOMAIN_AFFECTED_STAR_EXTRACTION':raise ValueError('affected-star status')
    b=load(BASE);o=load(OUT);d=load(SURFACE)
    p0=np.asarray(b['points_world_m'],np.float32);t0=np.asarray(b['tetrahedra'],np.int64);m0=np.asarray(b['material_indices'],np.int32);bf0=np.asarray(b['boundary_faces'],np.int64)
    p1=np.asarray(o['points_world_m'],np.float32);t1=np.asarray(o['tetrahedra'],np.int64);m1=np.asarray(o['material_indices'],np.int32);bf1=np.asarray(o['boundary_faces'],np.int64)
    node_map=np.asarray(o['input_node_to_output_node'],np.int64);node_inv=np.asarray(o['output_node_to_input_node'],np.int64)
    cell_map=np.asarray(o['input_cell_to_output_cell'],np.int64);cell_inv=np.asarray(o['output_cell_to_input_cell'],np.int64)
    if len(node_map)!=len(p0) or len(node_inv)!=len(p1) or len(cell_map)!=len(t0) or len(cell_inv)!=len(t1):raise ValueError('row-map dimensions')
    if not np.array_equal(np.asarray(o['retained_input_cell_rows'],np.int64),cell_inv):raise ValueError('retained cell rows and output inverse differ')
    if len(np.unique(cell_inv))!=len(cell_inv) or np.any(cell_inv<0) or np.any(cell_inv>=len(t0)):raise ValueError('output-to-input cell map not injective')
    if not np.array_equal(cell_map[cell_inv],np.arange(len(t1),dtype=np.int64)):raise ValueError('cell maps fail inverse roundtrip')
    if len(np.unique(node_inv))!=len(node_inv) or np.any(node_inv<0) or np.any(node_inv>=len(p0)):raise ValueError('output-to-input node map not injective')
    if not np.array_equal(node_map[node_inv],np.arange(len(p1),dtype=np.int64)):raise ValueError('node maps fail inverse roundtrip')
    if not np.array_equal(p1,p0[node_inv]):raise ValueError('output points differ from retained input Float32 point rows')
    used0=np.unique(t0)
    if np.any(node_map[used0]<0):raise ValueError('a referenced source point was dropped without quotient mapping')
    drop_to_keep={}
    volume_ops=transfer['surface']['operations']
    surface_ops=[json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    if len(volume_ops)!=307 or len(surface_ops)!=307:raise ValueError('307-operation ledger count')
    if len({int(x['drop_volume_node']) for x in volume_ops})!=307 or len({int(x['keep_volume_node']) for x in volume_ops})!=307:raise ValueError('edge endpoints are not unique across batch')
    if not np.array_equal(np.asarray(o['source_surface_contraction_ledger_rows'],np.int64),np.arange(307,dtype=np.int64)):raise ValueError('output contraction ledger rows')
    # Confirm each authorized D edge maps to the exact d4906 Float32 endpoints.
    sp=np.asarray(d.get('points_world_m_f32',d.get('points_world_m')),np.float32)
    coord_to_node={f32_key(pt):i for i,pt in enumerate(p0)}
    if len(coord_to_node)!=len(p0):raise ValueError('ambiguous input Float32 point identity')
    for i,(sop,vop) in enumerate(zip(surface_ops,volume_ops)):
        sd=int(sop['collapsed_input_point_row']);sk=int(sop['retained_input_point_row'])
        vd=int(vop['drop_volume_node']);vk=int(vop['keep_volume_node'])
        if int(vop['drop_D_row'])!=sd or int(vop['keep_D_row'])!=sk:raise ValueError(f'operation cross-ledger mismatch at {i}')
        if coord_to_node.get(f32_key(sp[sd]))!=vd or coord_to_node.get(f32_key(sp[sk]))!=vk:raise ValueError(f'D to Fat Float32 row crosswalk mismatch at op {i}')
        if node_map[vd]!=node_map[vk]:raise ValueError(f'quotient point map does not merge op endpoints at {i}')
        drop_to_keep[vd]=vk
    # Every referenced input point is either preserved as its same row or is an authorized collapsed row.
    for old in used0:
        new=int(node_map[old]);representative=int(node_inv[new])
        if old==representative:continue
        expected=drop_to_keep.get(int(old))
        while expected in drop_to_keep:expected=drop_to_keep[expected]
        if expected!=representative:raise ValueError(f'unapproved point merge {int(old)}->{representative}')
    # Reconstruct every surviving tetrahedron by the input-to-output vertex quotient.
    mapped_t=node_map[t0]
    if np.any(mapped_t<0):raise ValueError('referenced tetrahedron has unmapped node')
    sorted_map=np.sort(mapped_t,axis=1)
    deg=np.any(sorted_map[:,1:]==sorted_map[:,:-1],axis=1)
    if not np.array_equal(cell_map>=0,~deg):raise ValueError('retired cell map does not exactly match quotient-degenerate input tets')
    if int(np.count_nonzero(deg))!=len(t0)-len(t1):raise ValueError('degenerate cell count mismatch')
    expected_t=mapped_t[cell_inv]
    if not np.array_equal(np.sort(expected_t,axis=1),np.sort(t1,axis=1)):raise ValueError('output tetrahedra are not exactly the row-mapped input quotient')
    if not np.array_equal(m1,m0[cell_inv]):raise ValueError('material label changed on surviving cell rows')
    if len(m1)!=len(t1):raise ValueError('material row count')
    # Verify the per-operation affected-star sidecar directly from d4906 connectivity.
    flat=t0.reshape(-1);cell_rows=np.repeat(np.arange(len(t0),dtype=np.int64),4)
    order=np.argsort(flat,kind='stable');sorted_nodes=flat[order]
    node_offsets=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p0)),dtype=np.int64)]
    sorted_incident_cells=cell_rows[order]
    touched_union=set();star_points=set();star_cell_overlap_count=0;seen_cells=set()
    with np.load(STARS,allow_pickle=False) as z:star_arrays={k:np.asarray(z[k]) for k in z.files}
    offsets=star_arrays['per_operation_union_cell_star_offsets'];starrows=star_arrays['per_operation_union_cell_star_rows']
    all_drop=star_arrays['per_operation_drop_cell_star_rows'];dropoff=star_arrays['per_operation_drop_cell_star_offsets']
    all_keep=star_arrays['per_operation_keep_cell_star_rows'];keepoff=star_arrays['per_operation_keep_cell_star_offsets']
    for i,vop in enumerate(volume_ops):
        drop=int(vop['drop_volume_node']);keep=int(vop['keep_volume_node'])
        def incidence(node):
            # node IDs are dense and node_offsets is the CSR over sorted source connectivity.
            lo=int(node_offsets[node]);hi=int(node_offsets[node+1]);return np.sort(sorted_incident_cells[lo:hi])
        dr=incidence(drop);kr=incidence(keep);ur=np.union1d(dr,kr)
        rd=all_drop[dropoff[i]:dropoff[i+1]];rk=all_keep[keepoff[i]:keepoff[i+1]];ru=starrows[offsets[i]:offsets[i+1]]
        if not np.array_equal(dr,rd) or not np.array_equal(kr,rk) or not np.array_equal(ur,ru):raise ValueError(f'per-operation affected-star CSR mismatch at op {i}')
        tr=np.asarray(vop['touched_input_cell_rows'],np.int64)
        if not np.array_equal(dr,tr):raise ValueError(f'transfer drop-star cell ledger mismatch at op {i}')
        overlap=len(seen_cells.intersection(map(int,ur)))
        star_cell_overlap_count+=overlap;seen_cells.update(map(int,ur));touched_union.update(map(int,ur))
    if star_cell_overlap_count:raise ValueError(f'operation stars overlap in {star_cell_overlap_count} cells')
    affected_cells=star_arrays['affected_volume_cell_rows'];affected_points=star_arrays['affected_volume_point_rows']
    recomputed_affected_cells=np.asarray(sorted(touched_union),np.int64)
    recomputed_affected_points=np.unique(t0[recomputed_affected_cells].ravel())
    if not np.array_equal(affected_cells,recomputed_affected_cells):raise ValueError('affected-cell union CSR differs from independent star extraction')
    if not np.array_equal(affected_points,recomputed_affected_points):raise ValueError('affected-point list differs from independent union-star extraction')
    if np.any(node_map[affected_points]<0):raise ValueError('affected point has no output row')
    # Oriented boundary chain: reconstruct from all tets and compare both the published arrays and transformed d4906 boundary.
    bsrc=oriented_boundary(t0);bout=oriented_boundary(t1)
    if not np.array_equal(bsrc,face_signature(bf0)):raise ValueError('d4906 boundary_faces do not equal oriented tet boundary')
    if not np.array_equal(bout,face_signature(bf1)):raise ValueError('quotient boundary_faces do not equal oriented output tet boundary')
    mapped_b=node_map[bf0]
    sb=np.sort(mapped_b,axis=1);mapped_b=mapped_b[~np.any(sb[:,1:]==sb[:,:-1],axis=1)]
    if not np.array_equal(bout,face_signature(mapped_b)):raise ValueError('output oriented boundary differs from exact transformed source boundary')
    if not np.array_equal(bout,bsrc):
        # Difference is expected only from collapses; report symmetric changed boundary face count.
        pass
    # Float64 quality/volume replay on the emitted Float32 geometry.
    det0,q0,good0=measurements(p0,t0);det1,q1,good1=measurements(p1,t1)
    regressed=cell_inv[good0[cell_inv]&~good1]
    if len(regressed):raise ValueError(f'formerly-good retained cells regressed: {regressed[:12].tolist()}')
    bad0=int(np.count_nonzero(~good0));bad1=int(np.count_nonzero(~good1))
    if bad1>bad0:raise ValueError('global native geometry quality failures increased')
    # Exact integer-lattice orientation for every emitted Float32 tet.
    ip=f32_lattice_coords(p1);exact_bad=[]
    for i,tet in enumerate(t1):
        if exact_det(ip,tet)<=0:exact_bad.append(i)
    if exact_bad:raise ValueError(f'exact F32 nonpositive output tets: {exact_bad[:12]}')
    # Per-material cell volume and divergence-theorem boundary flux.
    mats=sorted(set(map(int,m0))|set(map(int,m1)));material=[]
    for mat in mats:
        mask0=(m0==mat);mask1=(m1==mat)
        v0=float(np.sum(det0[mask0]/6.0,dtype=np.float64));v1=float(np.sum(det1[mask1]/6.0,dtype=np.float64))
        bs0=oriented_boundary(t0[mask0]);bs1=oriented_boundary(t1[mask1])
        flux0=boundary_flux(p0,bs0);flux1=boundary_flux(p1,bs1)
        delta_cells=v1-v0;delta_boundary=flux1-flux0
        tol=2e-14*max(abs(v0),abs(v1),1.0)
        if abs(v0-flux0)>tol or abs(v1-flux1)>tol:raise ValueError(f'material {mat} tet volume disagrees with oriented boundary flux')
        if abs(delta_cells-delta_boundary)>tol:raise ValueError(f'material {mat} cell-volume delta does not match oriented boundary delta')
        declared=transfer['volume']['total_physical_volume_by_material_m3'].get(str(mat))
        if declared is None:raise ValueError(f'material {mat} absent from transfer receipt')
        if abs(float(declared['input_volume_m3'])-v0)>tol or abs(float(declared['output_volume_m3'])-v1)>tol or abs(float(declared['delta_m3'])-delta_cells)>tol:
            raise ValueError(f'material {mat} volume differs from declared transfer receipt')
        material.append({'material':mat,'input_tet_volume_m3':v0,'output_tet_volume_m3':v1,'tet_delta_m3':delta_cells,'input_oriented_boundary_flux_m3':flux0,'output_oriented_boundary_flux_m3':flux1,'boundary_delta_m3':delta_boundary,'declared_transfer_delta_m3':float(declared['delta_m3']),'absolute_consistency_tolerance_m3':tol})
    # Exact summary against the prior producer receipt.
    vol=transfer['volume']
    checks={
      'input_nodes':len(p0),'output_nodes':len(p1),'input_cells':len(t0),'output_cells':len(t1),
      'retired_degenerate_cells':int(np.count_nonzero(deg)),'input_boundary_faces':len(bf0),'output_boundary_faces':len(bf1),
      'contracted_node_count':len(drop_to_keep),'operation_count':len(volume_ops),
      'affected_union_cells':len(affected_cells),'affected_union_points':len(affected_points),
      'output_exact_integer_positive_cells':len(t1)-len(exact_bad),
      'input_native_quality_failures':bad0,'output_native_quality_failures':bad1,
      'formerly_good_retained_cell_regressions':len(regressed),
      'output_minimum_signed_det6_m3':float(det1.min()),'output_minimum_cell_volume_m3':float((det1/6).min()),
      'output_minimum_normalized_det':float(q1.min()),
      'producer_claimed_input_output_nodes':[int(vol['input_nodes']),int(vol['output_nodes'])],
      'producer_claimed_input_output_cells':[int(vol['input_cells']),int(vol['output_cells'])],
      'producer_claimed_input_output_quality_failures':[int(vol['input_native_quality_failures']),int(vol['output_native_quality_failures'])],
      'producer_boundary_equals_transformed_source':bool(vol['boundary_equals_exact_transformed_input_boundary']),
      'source_boundary_to_output_boundary_changed_oriented_faces':int(len(bout)-len(bsrc))
    }
    if checks['producer_claimed_input_output_nodes']!=[len(p0),len(p1)] or checks['producer_claimed_input_output_cells']!=[len(t0),len(t1)]:raise ValueError('volume report count mismatch')
    if checks['producer_claimed_input_output_quality_failures']!=[bad0,bad1]:raise ValueError('volume report quality counts mismatch')
    if len(regressed)!=int(vol['formerly_good_retained_cell_regressions']):raise ValueError('producer regression count mismatch')
    if len(exact_bad)!=0 or checks['producer_boundary_equals_transformed_source'] is not True:raise ValueError('independent exact output gate mismatch')
    report={
      'schema':'h706.d307-independent-volume-quotient-replay.v1','status':'PASS_INDEPENDENT_D307_VOLUME_MAP_BOUNDARY_QUALITY_MATERIAL_REPLAY',
      'physical_admission':False,'native_steps':0,'mass_mapping':'NOT_RUN',
      'source_pins':{str(Path(k)):sha(k) for k in [BASE,BASE_REPORT,OUT,VREPORT,SURFACE,SREPORT,LEDGER,STARS,STARS_REPORT]},
      'operation_crosswalk':{'count':len(volume_ops),'surface_ledger_matches_volume_ledger':True,'coordinate_crosswalk_exact_F32':True,'distinct_drop_nodes':len(drop_to_keep),'drop_keep_rows_approved':True},
      'row_map_replay':{'all_output_points_exact_input_rows':True,'all_output_tets_exact_point_quotient_of_input_rows':True,'retired_rows_exactly_degenerate_tets':True,'material_rows_preserved':True,'output_node_inverse_bijective':True,'output_cell_inverse_bijective':True},
      'oriented_boundary_replay':{'input_boundary_array_matches_input_tets':True,'output_boundary_array_matches_output_tets':True,'output_boundary_equals_transformed_input_boundary':True,'input_boundary_face_count':len(bf0),'output_boundary_face_count':len(bf1)},
      'affected_star_replay':{'status':'PASS','affected_cell_count':len(affected_cells),'affected_point_count':len(affected_points),'per_operation_drop_keep_union_CSR_matches_d4906_incidence':True,'transfer_touched_cell_rows_match':True,'operation_star_overlap_cells':star_cell_overlap_count},
      'quality_replay':{'metric':'signed_det6>0, det6/6>1e-18 m3, normalized_det6>=1e-5','all_output_float64_widened_F32_positive':bool(np.all(det1>0)),'all_output_exact_F32_integer_positive':True,'native_quality_failures_before_after':[bad0,bad1],'formerly_good_retained_cell_regressions':len(regressed),'minimum_cell_volume_m3':float((det1/6).min()),'minimum_normalized_det':float(q1.min())},
      'material_volume_and_boundary_flux':material,'counts_and_metrics':checks,
      'limitations':['D-only quotient of d4906; the full paired I/D/cavity boundary is not yet composed.','Mass mapping, native installation, native runtime, and physical admission were not run.'],
      'elapsed_s':time.monotonic()-started
    }
    outpath=OUTDIR/'report.json';outpath.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'report_path':str(outpath),'report_sha256':sha(outpath),'elapsed_s':report['elapsed_s'],'counts':checks,'material':material},indent=2))

if __name__=='__main__':run()
