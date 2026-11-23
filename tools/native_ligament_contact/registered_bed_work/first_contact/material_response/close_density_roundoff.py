"""Correct roundoff-sized density overshoots by exchanging mass along an existing source carrier."""
from fractions import Fraction
import json
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import sha,tetgen_positions


def volume(vertices):
    x = [[Fraction(float(v)) for v in row] for row in vertices]
    a,b,c = [[p[i]-x[0][i] for i in range(3)] for p in x[1:]]
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if det <= 0: raise ValueError('Inverted material')
    return float(det/6)


def run(mesh,flows,audit,carriers,out):
    fail=json.loads(audit.read_bytes())
    t=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    xyz=tetgen_positions(mesh/'mesh.node').astype('f4').astype('f8')
    reference=np.fromfile(mesh/'cell-mass.f64',dtype='f8')
    out.mkdir()
    receipt={}
    for mat,tag,lo,hi in ((15,'SKIN',900,1100),(16,'FAT',850,950)):
        src=flows/(tag+'-physical-flow.npz')
        if sha(src)!=fail[tag]['flow_sha256']:raise ValueError('Audit describes another allocation')
        data=np.load(src);q,g,m=(data[k] for k in ('carrier','role_cell','mass'))
        role=np.flatnonzero(t[:,4]==mat)
        prior=np.load(carriers/(tag+'-candidate-flow.npz'))
        typed=prior['eligible']
        exceptional=prior['role_cell'][~typed].astype('i8')*int(q.max()+1)+prior['carrier'][~typed]
        allowed=~np.isin(g.astype('i8')*int(q.max()+1)+q,exceptional)
        masses=np.bincount(g,weights=m,minlength=len(role))
        bad=np.searchsorted(role,fail[tag]['density_failures_global'])
        forbidden=np.isin(g,bad)
        changes=[]
        for row in bad:
            V=volume(xyz[t[role[row],:4]-1])
            rho=masses[row]/V
            bound=lo if rho<lo else hi
            # Tiny guard for summation across recipients, below the original
            # target-mass comparison tolerance. Preserve each source debit.
            desired=(bound*(1+2e-13 if rho<lo else 1-2e-13))*V
            delta=desired-masses[row]
            ix=np.flatnonzero((g==row)&allowed&(m>2*abs(delta)))
            resolved=False
            for i in ix[np.argsort(-m[ix])]:
                parents=np.flatnonzero((q==q[i])&~forbidden&allowed&(m>2*abs(delta)))
                for j in parents[np.argsort(-m[parents])]:
                    neighbor=g[j];nv=volume(xyz[t[role[neighbor],:4]-1])
                    after=(masses[neighbor]-delta)/nv
                    if not lo-16*np.finfo(float).eps*hi <= after <= hi+16*np.finfo(float).eps*hi:continue
                    if abs(delta)>1e-10*min(reference[role[row]],reference[role[neighbor]]):continue
                    m[i]+=delta;m[j]-=delta;masses[neighbor]-=delta;masses[row]+=delta
                    changes.append(dict(global_recipient=int(role[row]),same_carrier=int(q[i]),
                                   other_global_recipient=int(role[neighbor]),mass_exchange_kg=float(delta)))
                    resolved=True
                    break
                if resolved:break
            if not resolved:raise ValueError(f'No conservative roundoff exchange for {tag} row {row}')
        dest=out/(tag+'-physical-flow.npz')
        np.savez_compressed(dest,carrier=q,role_cell=g,mass=m)
        receipt[tag]=dict(rebalances=changes,original=sha(src),result=sha(dest))
        print(tag,len(changes),flush=True)
    (out/'density-exchanges.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':run(*(Path(arg).resolve() for arg in sys.argv[1:]))
