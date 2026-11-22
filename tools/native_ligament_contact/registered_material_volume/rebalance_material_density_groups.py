"""Move mass only on previously supported carrier routes and close the changed cell groups."""
import argparse,json
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha

def main():
    parser=argparse.ArgumentParser()
    for k in ('parent','dense','volumes','elements','geometry','output'):parser.add_argument(k,type=Path)
    a=parser.parse_args();out=a.output;out.mkdir(exist_ok=True)
    name='current-density-closed-flow.npz';src=a.parent/name;r=json.loads((a.parent/'mass-transfer.json').read_text())
    assert r['flow_sha256']==sha(src) and r['source_ledger_compaction']['source_flow_sha256']==sha(a.dense/name)
    f=dict(np.load(src));dense=np.load(a.dense/name);c=f['carrier_row'];g=f['group'];cg=f['group_for_cell']
    assert np.array_equal(c,dense['carrier_row']) and np.array_equal(g,dense['group'])
    v=np.fromfile(a.volumes,dtype='f8');assert len(v)==len(cg) and (v>1e-18).all()
    with a.elements.open() as h:n=int(next(h).split()[0]);e=np.loadtxt(h,max_rows=n,usecols=(1,2,3,4,5),dtype='i4')
    _,first=np.unique(cg,return_index=True);role=e[:,4];assert np.array_equal(role[first][cg],role)
    lo=np.where(role[first]==15,900.,850.);hi=np.where(role[first]==15,1100.,950.);gv=np.bincount(cg,weights=v)
    m=f['mass_flow_kg'].copy();parent_m=m.copy();gm=np.bincount(g,weights=m)
    L=lo*gv;H=hi*gv;inset=np.maximum(1e-12*gm,16*np.spacing(gm))
    assert np.all(H-L>2*inset)
    allowed=np.flatnonzero((dense['mass_flow_kg']>0)&~f['archived_compensation'])
    source=defaultdict(list);cellgroup=defaultdict(list)
    for edge in allowed:source[int(c[edge])].append(int(edge));cellgroup[int(g[edge])].append(int(edge))
    history=[]
    for _ in range(200):
        gm=np.bincount(g,weights=m);avg=gm/gv;bad=np.flatnonzero((avg<lo)|(avg>hi))
        if not len(bad):break
        j=int(bad[np.argmax(np.maximum(L[bad]-gm[bad],gm[bad]-H[bad]))]);pull=avg[j]<lo[j]
        change=(L[j]+inset[j]-gm[j]) if pull else (gm[j]-H[j]+inset[j])
        prev={j:None};queue=deque([j]);found=None
        while queue and found is None:
            here=queue.popleft()
            for ei in cellgroup[here]:
                if not pull and m[ei]<=0:continue
                for ej in source[int(c[ei])]:
                    peer=int(g[ej]);debit,credit=(ej,ei) if pull else (ei,ej)
                    if peer in prev or m[debit]<=0:continue
                    prev[peer]=(here,debit,credit)
                    slack=gm[peer]-L[peer]-inset[peer] if pull else H[peer]-gm[peer]-inset[peer]
                    if slack>max(16*np.spacing(gm[peer]),1e-22):found=(peer,slack);break
                    queue.append(peer)
                if found:break
        if found is None:raise ValueError(('No conservative path for recipient',j,change))
        k,slack=found;path=[]
        while k!=j:
            k,ei,ej=prev[k];path.append((ei,ej))
        amount=min(change,slack,min(m[ei] for ei,ej in path))
        for ei,ej in path:m[ei]-=amount;m[ej]+=amount
        history.append(dict(recipient=j,mass_kg=float(amount),source_carrier_edges=path))
    gm=np.bincount(g,weights=m);avg=gm/gv;assert np.all((avg>=lo)&(avg<=hi))
    assert np.all(m>=0) and np.array_equal(m[f['archived_compensation']],parent_m[f['archived_compensation']])
    groups=np.unique(g[m!=parent_m])
    em=f['element_mass_kg'].copy();actual=em*(gm/np.bincount(cg,weights=em))[cg]/v
    groups=np.union1d(groups,cg[(actual<lo[cg])|(actual>hi[cg])|(em/v<lo[cg])|(em/v>hi[cg])])
    use=np.isin(cg,groups);em[use]=v[use]*avg[cg[use]];actual=em*(gm/np.bincount(cg,weights=em))[cg]/v
    if not np.all((actual>=lo[cg])&(actual<=hi[cg])&(em/v>=lo[cg])&(em/v<=hi[cg])):raise ValueError('Native density margin exhausted')
    error=np.abs(np.bincount(c,weights=m)-np.bincount(c,weights=parent_m)).max()
    if error>2e-17:raise ValueError('Carrier quota drift')
    f['mass_flow_kg']=m;f['element_mass_kg']=em;f['nodal_mass_kg']=sum(np.bincount(e[:,i]-1,weights=em*.25,minlength=len(f['nodal_mass_kg'])) for i in range(4))
    dst=out/name;np.savez_compressed(dst,**f)
    changed=np.flatnonzero(parent_m!=m)
    result=dict(geometry_sha256=sha(a.geometry),parent_flow_sha256=sha(src),flow_sha256=sha(dst),
        admitted_route_distance_m=r['admitted_route_distance_m'],solver_output=f'augmentations={len(history)}',
        used_new_typed_routes=int(np.count_nonzero((parent_m==0)&(m>0))),changed_edges=len(changed),
        archived_compensation_unchanged=True,maximum_carrier_mass_change_kg=float(error),
        corrected_groups=groups.tolist(),corrected_cell_count=int(use.sum()),native_volume_sha256=sha(a.volumes),
        density_outliers=int(np.count_nonzero((actual<lo[cg])|(actual>hi[cg])|(em/v<lo[cg])|(em/v>hi[cg]))),
        total_mass_kg=float(em.sum()),minimum_density_kg_m3=float(actual.min()),maximum_density_kg_m3=float(actual.max()),
        transfers=history)
    (out/'mass-transfer.json').write_text(json.dumps(result,indent=2)+'\n')
    print('groups',len(groups),'cells',int(use.sum()),'mass transfers',len(history),'carrier error',error)
if __name__=='__main__':main()
