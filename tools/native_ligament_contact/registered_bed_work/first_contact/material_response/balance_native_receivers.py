"""Move only native density roundoff through the existing carrier-parent-receiver graph."""
from pathlib import Path
import sys,json
from collections import deque
import numpy as np
from compile_connected_joint import digest
from supported_material_admission import geom


def run(transfer,flow,carriers,mesh,domain,native,witness,out):
    rows=np.atleast_2d(np.loadtxt(witness));start=int(np.load(native/'material-source-indices.npz')['child_begin'])
    z=np.load(domain);ov=np.load(flow/'ancestry/overlap.npz')
    fmt=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4');x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3));out.mkdir();summary={}
    for role,tag,lo,hi in [(15,'SKIN',900.,1100.),(16,'FAT',850.,950.)]:
        a=np.load(transfer/(tag+'-transport.npz'));src=np.load(carriers/'flows'/(tag+'-physical-flow.npz'))
        before=json.loads((transfer/(tag+'-transport.json')).read_text());assert before['final']['passed']
        bank=np.flatnonzero(fmt[:,5]==role);oldparent=bank[src['role_cell']];inside=np.isin(oldparent,ov['parent_ids'])
        qq=a['quadrature_group'];external=a['external'];parents=np.unique(oldparent[inside]);children=a['child_ids']
        P=int(qq.max())+1;E=P+len(parents);C=E+len(external);nodes=C+len(children)
        if np.any(oldparent[~inside]!=external[np.searchsorted(external,oldparent[~inside])]):raise ValueError('Wrong external destinations')
        sink=np.empty(len(qq),int);sink[inside]=P+np.searchsorted(parents,oldparent[inside]);sink[~inside]=E+np.searchsorted(external,oldparent[~inside])
        u=np.r_[qq,P+np.searchsorted(parents,a['parent'])];v=np.r_[sink,C+np.searchsorted(children,a['child'])]
        nu=len(qq);old=np.r_[a['upstream_kg'],a['downstream_kg']];f=old.copy()
        accr=np.bincount(v,weights=old,minlength=nodes)[E:]
        mass=np.r_[a['external_mass_kg'],a['child_mass_kg']];reserve=mass-accr
        volume=np.r_[geom(x,fmt[external,1:5]-1)[0],ov['child_volume_mm3'][children]*1.e-9]
        minimum=(lo+5e-8)*volume;maximum=(hi-5e-8)*volume
        w=rows[rows[:,1]==role];bad=set()
        for ir,_,rho,vol,m in w:
            j=int(ir)-start;k=len(external)+int(np.searchsorted(children,j))
            if j<0 or children[k-len(external)]!=j or abs(m-mass[k])>2e-14*m:raise ValueError('Unmatched recipient')
            minimum[k]=max(minimum[k],(lo+5e-8)*vol);maximum[k]=min(maximum[k],(hi-5e-8)*vol);bad.add(k+E)
        # One CSR adjacency holds both residual directions. Positive traversal increases that edge.
        ids=np.arange(len(f));here=np.r_[u,v];to=np.r_[v,u];edge=np.r_[ids,ids];sense=np.r_[np.ones(len(f),'i1'),-np.ones(len(f),'i1')]
        order=np.argsort(here,kind='stable');ptr=np.r_[0,np.cumsum(np.bincount(here,minlength=nodes))]
        to,edge,sense=to[order],edge[order],sense[order];prev=np.full(nodes,-1,int);visit=np.zeros(nodes,int);paths=[]
        for gen,root in enumerate(sorted(bad),1):
            ri=root-E
            add=mass[ri]<minimum[ri];amount=(minimum[ri]-mass[ri]) if add else mass[ri]-maximum[ri]
            if amount<=0:continue
            queue=deque([root]);visit[root]=gen;peer=None
            while queue and peer is None:
                current=queue.popleft();ss=slice(ptr[current],ptr[current+1])
                adj=np.arange(ptr[current],ptr[current+1]);ds=sense[ss]*(-1 if add else 1)
                usable=(visit[to[ss]]!=gen)&((ds>0)|(f[edge[ss]]>=amount))
                for k in adj[usable]:
                    n=to[k]
                    if visit[n]==gen:continue
                    visit[n]=gen;prev[n]=k
                    if n>=E and n not in bad:
                        capacity=mass[n-E]-minimum[n-E] if add else maximum[n-E]-mass[n-E]
                        if capacity>=2*amount:peer=n;break
                    queue.append(n)
            if peer is None:raise ValueError(f'No admissible carrier exchange for {tag} {root}')
            n=peer;path=[]
            while n!=root:
                k=prev[n];e=edge[k];ds=int(sense[k])*(-1 if add else 1)
                f[e]+=ds*amount;path.append([int(e),ds]);n=u[e] if sense[k]>0 else v[e]
            mass[ri]+=amount if add else -amount;mass[peer-E]-=amount if add else -amount
            paths.append(dict(target=int(ri),peer=int(peer-E),kg=float(amount),edges=path))
        if f.min()<0.:raise ValueError('Negative flow')
        src0=np.bincount(qq,weights=old[:nu]);src1=np.bincount(qq,weights=f[:nu]);source_err=float(np.max(abs(src1-src0)/np.maximum(src0,1e-25)))
        pin=np.bincount(sink,weights=f[:nu],minlength=nodes)[P:E];pout=np.bincount(u[nu:],weights=f[nu:],minlength=nodes)[P:E]
        parent_err=float(np.max(abs(pin-pout)));rebuild=reserve+np.bincount(v,weights=f,minlength=nodes)[E:]
        diff=rebuild-np.r_[a['external_mass_kg'],a['child_mass_kg']]
        pos=np.vstack((x[fmt[external,1:5]-1].mean(1),z['x'].astype(float)[z['t'][children]].mean(1)))
        stat=dict(before['final']);stat['max_source_relative_error']+=source_err;stat['max_parent_error_kg']=parent_err
        stat['mass_error_kg']+=float(diff.sum());stat['first_moment_kg_m']=(np.asarray(stat['first_moment_kg_m'])+diff@pos).tolist()
        rho=rebuild/volume;stat['density_range']=[float(rho.min()),float(rho.max())]
        stat['passed']=bool(stat['max_source_relative_error']<1e-9 and parent_err<1e-14 and abs(stat['mass_error_kg'])<1e-11 and max(map(abs,stat['first_moment_kg_m']))<1e-11 and rho.min()>=lo-1e-8 and rho.max()<=hi+1e-8)
        if not stat['passed']:raise ValueError(f'Unqualified receiver exchange {stat}')
        vals=dict(upstream_kg=f[:nu],downstream_kg=f[nu:],external_mass_kg=rebuild[:len(external)],child_mass_kg=rebuild[len(external):])
        np.savez_compressed(out/(tag+'-transport.npz'),**{k:vals.get(k,a[k]) for k in a.files})
        evidence=dict(parent=digest(transfer/(tag+'-transport.npz')),witness=digest(witness),paths=paths)
        (out/(tag+'-transport.json')).write_text(json.dumps(dict(final=stat,native_exchange=evidence),indent=2)+'\n')
        summary[tag]=dict(targets=len(paths),total_transfer_kg=sum(p['kg'] for p in paths),longest_path=max([0,*[len(p['edges']) for p in paths]]))
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
