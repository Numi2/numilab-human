"""Reconcile proven one-hop material ancestry and bounded historical exceptions on current positive flows."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from compile_connected_joint import digest
from register_material_surfaces import pinned
from solve_connected_carrier_flow import solve
from exchange_material_credits import connected_flow,exchange

def run(control,flows,exe,out):
    e=json.loads(control.read_bytes())['environment'];report=json.loads((flows/'result.json').read_bytes())
    pre='NUMI_H706_FRACTIONAL_';out.mkdir();result={};pinned_path=lambda k:Path(e[k])
    for what in ['SOURCE330_TETS','SOURCE330_NODES']:
        if digest(pinned_path(pre+what))!=e[pre+what+'_SHA256']:raise ValueError('Parent registry mismatch')
    vertices=np.loadtxt(e[pre+'SOURCE330_NODES'],skiprows=1,delimiter=',',usecols=(5,6,7))
    parents=np.loadtxt(e[pre+'SOURCE330_TETS'],skiprows=1,delimiter=',',usecols=(3,4,5,6),dtype='i4')
    labels=np.loadtxt(pinned(e,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256'),skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4')
    t=np.loadtxt(pinned(e,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'),skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    x=np.loadtxt(pinned(e,'TETGEN_NODE','TETGEN_NODE_SHA256'),skiprows=1,usecols=(1,2,3),dtype='f4').astype('f8')
    radius=float(e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_MAXIMUM_MASS_TRANSPORT_DISTANCE_M'])
    for tag,m in [('SKIN',15),('FAT',16)]:
        flow=flows/(tag+'-physical-flow.npz')
        if digest(flow)!=report[tag]['flow_sha256']:raise ValueError('Flow identity changed')
        f=dict(np.load(flow));q=f['carrier'];c=f['role_cell'];w=f['mass'];cells=np.flatnonzero(t[:,4]==m);tt=t[cells,:4]-1;pos=x[tt].mean(1)
        for k in ['CARRIERS_','P1_']:
            if digest(pinned_path(pre+k+tag+'_NPZ'))!=e[pre+k+tag+'_SHA256']:raise ValueError('Wrong support')
        p=dict(np.load(e[pre+'P1_'+tag+'_NPZ']));v=dict(np.load(e[pre+'CARRIERS_'+tag+'_NPZ']))
        par=np.repeat(v['global_source330_tet_rows'],np.diff(v['parent_offsets']));reg=labels[par,2];qp=np.einsum('ij,ijk->ik',v['carrier_barycentric'],vertices[parents[par]])
        off=p['cell_admissible_region_offsets'];regions=p['cell_admissible_region_ids']
        def allowed(qq,rr):
            answer=((qp[qq]-pos[rr])**2).sum(1)<=radius**2;groups=reg[qq]
            for group in np.unique(groups):
                ix=np.flatnonzero(groups==group);valid=np.searchsorted(off[1:],np.flatnonzero(regions==group),side='right')
                answer[ix]&=np.isin(rr[ix],valid)
            return answer
        normal=allowed(q,c);exception=np.zeros(len(q),bool)
        if tag=='SKIN':
            pot=(~normal)&np.isin(par[q],v['regional_compensation_parent_rows'])&np.isin(cells[c],v['regional_compensation_child_rows'])
            ii=np.flatnonzero(pot);exception[ii]=np.linalg.norm(qp[q[ii]]-pos[c[ii]],axis=1)<=v['regional_compensation_limits'][1]
            if w[exception].sum()>v['regional_compensation_limits'][0]:raise ValueError('Compensation capacity exceeded')
        bad=~(normal|exception);offenders=np.unique(c[bad]);history=[]
        print(tag,'bad',bad.sum(),'kg',w[bad].sum(),'cells',len(offenders),'historical exception kg',w[exception].sum(),flush=True)
        details=[]
        for dest in offenders:
            inds=np.flatnonzero((c==dest)&bad);sources=np.unique(q[inds]);groups=regions[off[dest]:off[dest+1]]
            candidates=np.flatnonzero(np.isin(reg,groups));dist=np.linalg.norm(qp[candidates]-pos[dest],axis=1);eligible_q=candidates[dist<=radius]
            details.append(dict(role_cell=int(dest),cell=int(cells[dest]),kg=float(w[inds].sum()),ids=sources.tolist(),source_regions=reg[sources].tolist(),target_regions=groups.tolist(),eligible_carriers=len(eligible_q),closest_m=float(dist.min()),point=pos[dest].tolist(),eligible_capacity_kg=float(np.sum(w[np.isin(q,eligible_q)]))))
        (out/(tag+'-problem.json')).write_text(json.dumps(details,indent=2)+'\n')
        if len(offenders):
            # Stay within the physical mass radius and the retained local registration diameter.
            if tag=='SKIN':
                attempt=exchange(q,c,w,normal,bad,tt,pos,allowed,radius)
                if attempt[3]['unmatched']:q,c,w,rec=connected_flow(exe,q,c,w,normal,bad,tt,pos,qp,reg,off,regions,radius,allowed)
                else:q,c,w,rec=attempt
            else:q,c,w,rec=connected_flow(exe,q,c,w,normal,bad,tt,pos,qp,reg,off,regions,radius,allowed)
            history.append(rec);print(tag,rec,flush=True)
            if rec['unmatched']:
                (out/(tag+'-unmatched.json')).write_text(json.dumps(rec)+'\n');continue
        source=np.bincount(q,weights=w,minlength=len(par));prior=np.bincount(f['carrier'],weights=f['mass'],minlength=len(par));recv=np.bincount(c,weights=w,minlength=len(cells));previous=np.bincount(f['role_cell'],weights=f['mass'],minlength=len(cells))
        if np.any(abs(source-prior)>np.maximum(prior*3e-12,3e-19)) or np.any(abs(recv-previous)>np.maximum(previous*3e-12,3e-19)):raise ValueError('Cell or source mass changed')
        checked=allowed(q,c);exceptions=~checked
        if np.any(exceptions) and (tag!='SKIN' or not np.all(np.isin(par[q[exceptions]],v['regional_compensation_parent_rows']) & np.isin(cells[c[exceptions]],v['regional_compensation_child_rows']))):raise ValueError('Unauthorized exception')
        dest=out/(tag+'-physical-flow.npz');np.savez_compressed(dest,carrier=q,role_cell=c,mass=w)
        result[tag]=dict(flow_sha256=digest(dest),native_source_edges=len(q),bad_edges_before=int(bad.sum()),previously_unsupported_kg=float(f['mass'][bad].sum()),unsupported_edges_after=0,retained_exception_kg=float(w[exceptions].sum()),original_flow=digest(flow),maximum_target_error_kg=float(max(abs(recv-previous))),maximum_source_error_kg=float(max(abs(source-prior))),solves=history)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
