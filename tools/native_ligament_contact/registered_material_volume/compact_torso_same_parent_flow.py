"""Cancel positive same-parent transport cycles without changing a recipient's parent contributions."""
import argparse,json
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

def main():
    p=argparse.ArgumentParser()
    for k in ('source','geometry','volumes','out'):p.add_argument(k,type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);r=json.load((args.source/'mass-transfer.json').open());packet=args.source/'current-density-closed-flow.npz'
    assert sha(packet)==r['flow_sha256'] and sha(args.volumes)==r['native_reference_volumes_sha256']
    f=dict(np.load(packet));ids=f['carrier_row'];g=f['group'];par=f['source_parent'];base=f['mass_flow_kg'];m=base.copy()
    # The optimization graph uses only existing positive ordinary routes.
    pos=np.flatnonzero((m>0)&~f['archived_compensation']);order=pos[np.lexsort((g[pos],ids[pos],par[ids[pos]]))]
    parent_id=par[ids[order]];start=np.r_[0,np.flatnonzero(np.diff(parent_id))+1,len(order)]
    cellg=f['group_for_cell'];weights=np.bincount(cellg);removed=cycles=0
    for a,b in zip(start[:-1],start[1:]):
        ii=order[a:b];by={}
        for e in ii:by.setdefault(int(ids[e]),{})[int(g[e])]=int(e)
        keys=sorted(by)
        for k,c in enumerate(keys):
            ca=by[c]
            for v in keys[k+1:]:
                cb=by[v];common=sorted((j for j in ca.keys()&cb.keys() if m[ca[j]]>0 and m[cb[j]]>0),key=lambda j:(-weights[j],j))
                # A bipartite four-cycle exchanges mass within the same
                # anatomical parent. Transport into each recipient is unchanged.
                while len(common)>=2:
                    j=common.pop();l=common[-1]
                    aj,bj,al,bl=ca[j],cb[j],ca[l],cb[l]
                    choices=[(min(m[aj],m[bl]),aj,bl,bj,al),(min(m[bj],m[al]),bj,al,aj,bl)]
                    def choice_key(item):
                        amount,*ed=item
                        saving=sum(int(weights[g[e]]) for e in ed[:2] if m[e]==amount)
                        return (-saving,amount,ed)
                    amt,ma,mb,pa,pb=min(choices,key=choice_key)
                    assert amt>0
                    m[ma]-=amt;m[mb]-=amt;m[pa]+=amt;m[pb]+=amt;cycles+=1
                    for e in (ma,mb):
                        if m[e]==0.:removed+=int(weights[g[e]])
                    if m[al]==0 or m[bl]==0:common.pop()
                    elif m[aj]>0 and m[bj]>0:common.append(j)
    assert np.all(np.isfinite(m)&(m>=0)) and np.array_equal(m[f['archived_compensation']],base[f['archived_compensation']])
    parent_group=par[ids].astype('i8')*len(weights)+g
    _,pi=np.unique(parent_group,return_inverse=True);pg=float(np.max(abs(np.bincount(pi,weights=base)-np.bincount(pi,weights=m))))
    ce=float(np.max(abs(np.bincount(ids,weights=base)-np.bincount(ids,weights=m))));ge=float(np.max(abs(np.bincount(g,weights=base)-np.bincount(g,weights=m))))
    assert max(pg,ce,ge)<1e-14
    f['mass_flow_kg']=m
    with args.geometry.open() as h:nt=int(next(h).split()[0]);e=np.loadtxt(h,max_rows=nt,usecols=(1,2,3,4,5),dtype='i4')
    vol=np.fromfile(args.volumes,dtype='f8');low=np.where(e[:,4]==15,900.,850.);high=np.where(e[:,4]==15,1100.,950.)
    assigned=np.bincount(g,weights=m);em=f['element_mass_kg'].copy();rho=em*(assigned/np.bincount(cellg,weights=em))[cellg]/vol
    bad=np.unique(cellg[(rho<low)|(rho>high)]);sel=np.isin(cellg,bad);gv=np.bincount(cellg,weights=vol)
    em[sel]=vol[sel]*(assigned/gv)[cellg[sel]]
    rho=em*(assigned/np.bincount(cellg,weights=em))[cellg]/vol;assert np.all((rho>=low)&(rho<=high))
    f['element_mass_kg']=em;f['nodal_mass_kg']=sum(np.bincount(e[:,i]-1,weights=em*.25,minlength=len(f['nodal_mass_kg'])) for i in range(4))
    dest=args.out/packet.name;np.savez_compressed(dest,**f)
    report=dict(source_flow_sha256=sha(packet),cycles=cycles,positive_input_edges=int(sum(base>0)),positive_output_edges=int(sum(m>0)),expanded_rows_removed=removed,maximum_parent_recipient_change_kg=pg,maximum_carrier_change_kg=ce,maximum_recipient_change_kg=ge,additional_active_edges=0,archived_compensation_amounts_identical=True,cell_masses_changed=int(sum(em!=np.load(packet)['element_mass_kg'])),minimum_actual_density_margin_kg_m3=float(np.min(np.minimum(rho-low,high-rho))))
    r.update(flow_sha256=sha(dest),source_ledger_compaction=report,result_mass_kg=float(em.sum()),nodal_mass_kg=float(f['nodal_mass_kg'].sum()))
    (args.out/'mass-transfer.json').write_text(json.dumps(r,indent=2)+'\n');print(report,flush=True)
if __name__=='__main__':main()
