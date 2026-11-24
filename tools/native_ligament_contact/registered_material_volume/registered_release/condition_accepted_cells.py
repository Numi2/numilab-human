"""Relocate material-interior sampling sites through one barycentric map in every pose."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
from supported_material_admission import geom
from compile_connected_joint import digest
from certify_interior_motion import response
from transfer_cavities import intersection_fraction

def build(vol,loaded,accepted,held,quality,material,out):
    data=json.loads(quality.read_bytes())
    if digest(accepted)!=data['geometry_sha256'] or digest(vol/'node-mass.f64')!=data['source_mass_sha256']:raise ValueError('Stale endpoints')
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    positions=[read(f)for f in (vol/'mesh.node',loaded,accepted,held)]
    raw=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');cells=raw[:,:4]-1
    coeff=json.loads(material.read_bytes())['materials'];masses=np.fromfile(vol/'cell-mass.f64','<f8')
    nodes=np.unique([n for row in data['rows'] for n in row['homogeneous_free_corners']]);sel=np.flatnonzero(np.isin(cells,nodes).any(1))
    labels=np.loadtxt(vol/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
    proposals=[];tested=0;geometrical=0;rejections={};allbest=[]
    for n in nodes:
        rows=sel[np.any(cells[sel]==n,1)];ids,tc=np.unique(cells[rows],return_inverse=True);tc=tc.reshape((-1,4))
        if len(set(raw[rows,4]))!=1:raise ValueError('Material interface')
        coords=[x[ids]for x in positions];slot=np.searchsorted(ids,n);role=int(raw[rows[0],4]);par=coeff[str(role)]['registered'];props=(par['source_mu_pa'],par['source_logJ_coefficient_pa'])
        previous=[geom(p,tc)for p in coords];oldbad=np.flatnonzero(previous[2][1]<1e-5)
        base_mech=[response(coords[0],p,tc,*props) for p in coords[1:]]
        best=None
        for parent in tc:
            neigh=parent[parent!=slot];dv=[];tests=[];rhs=[]
            for k,(p,(v,q)) in enumerate(zip(coords[:3],previous[:3])):
                slope=[]
                for m in neigh:
                    g=p.copy();g[slot]=p[m];slope.append(geom(g,tc)[0]-v)
                slope=np.stack(slope,axis=1);dv.append(slope)
                # Existing long edges set the target scale. Re-evaluate all exact edge lengths after motion.
                scale=v/q; goal=1.01e-5*scale
                tests.extend(-slope/scale[:,None]*1e5);rhs.extend((v-goal)/scale*1e5)
            for k in (1,2):
                for sign,threshold in ((1,.35),(-1,2.5)):
                    s=sign*(dv[k]-threshold*dv[0]);m=sign*(previous[k][0]-threshold*previous[0][0])
                    tests.extend(-s/previous[0][0][:,None]);rhs.extend(m/previous[0][0])
            lens=np.linalg.norm(coords[0][neigh]-coords[0][slot],axis=1)
            # Use the shortest admissible displacement instead of maximizing the slack on healthy cells.
            lp=linprog(lens/max(lens),A_ub=np.vstack((tests,np.ones((1,3)))),b_ub=np.r_[rhs,.5],bounds=(0,.5),method='highs')
            tested+=1
            if not lp.success:continue
            for factor in (1.,1.05,1.2,1.5):
                w=lp.x*factor
                if sum(w)>.5:continue
                B=np.eye(len(ids));B[slot,slot]=1-sum(w);B[slot,neigh]=w
                moved=[B@p for p in coords]
                # The reference uses one Float32 word. Matter stores the loaded
                # displacement in two, including during its geometry checks.
                reference = moved[0][slot].astype('f4').astype('f8')
                displacement = moved[1][slot] - reference
                high = displacement.astype('f4').astype('f8')
                low = (displacement - high).astype('f4').astype('f8')
                moved[0][slot] = reference
                moved[1][slot] = reference + (high + low)
                after=[geom(p,tc) for p in moved]
                if any((v<=1e-18).any() or (q<1e-5).any() for v,q in after[:3]):continue
                if np.any((after[3][1]<1e-5)&(previous[3][1]>=1e-5)) or np.any(after[3][0]<=1e-18):continue
                if any(((v/after[0][0]<=.35)|(v/after[0][0]>=2.5)).any() for v,q in after[1:3]):continue
                geometrical+=1;errors=[]
                for p,(oldE,f,_) in zip(moved[1:],base_mech):
                    e,ff,_=response(moved[0],p,tc,*props)
                    errors.append(dict(force=float(np.linalg.norm(B.T@ff-f)/max(np.linalg.norm(f),1e-14)),energy=float(abs(e-oldE)/max(abs(oldE),1e-18))))
                if any(e['force']>.01 or e['energy']>.05 for e in errors):
                    rejections['force_or_energy']=rejections.get('force_or_energy',0)+1;continue
                cand=dict(vertex=int(n),material=role,affected_cells=rows.tolist(),resolved_cells=rows[oldbad].tolist(),neighbors=ids[neigh].tolist(),weights=w.tolist(),errors=errors,reference_shift_um=float(np.linalg.norm(moved[0][slot]-coords[0][slot])*1e6),positions=[p[slot].tolist()for p in moved])
                if best is None or best['reference_shift_um']>cand['reference_shift_um']:best=cand
        if best:allbest.append(best)
    used=set()
    for r in sorted(allbest,key=lambda x:(-len(x['resolved_cells']),x['reference_shift_um'])):
        rows=np.array(r['affected_cells'])
        if not used.isdisjoint(rows):continue
        ids,tc=np.unique(cells[rows],return_inverse=True);tc=tc.reshape(-1,4);node=np.searchsorted(ids,r['vertex']);X=positions[0][ids];verts=np.vstack([X,r['positions'][0]])
        new=tc.copy();new[new==node]=len(X)
        weights=np.asarray([[intersection_fraction(a,b,verts)for b in new]for a in tc])
        V=geom(X,tc)[0];vn=geom(verts,new)[0]
        rowerr=max(abs(weights.sum(1)-1));colerr=max(abs(weights.T@V/vn-1))
        if max(rowerr,colerr)>3e-9:
            rejections['source_overlay']=rejections.get('source_overlay',0)+1;continue
        weights/=weights.sum(1)[:,None];mass=weights.T@masses[rows];delta=float(abs(sum(mass)-sum(masses[rows])))
        centers=verts[new].mean(1)-X[0];oldcenters=X[tc].mean(1)-X[0]
        first=float(np.linalg.norm(mass@centers-masses[rows]@oldcenters))
        # Keep every old source credit as a fractional transfer entry.
        i,k=np.nonzero(weights)
        r.update(overlap_rows=rows[i].tolist(),overlap_destinations=rows[k].tolist(),fractions=weights[i,k].tolist(),donor_identities=labels[rows,1:].tolist(),mass_kg=mass.tolist(),mass_error_kg=delta,first_moment_error_kg_m=first,overlap_error=max(rowerr,colerr),density_kg_m3=[float(min(mass/vn)),float(max(mass/vn))])
        if delta>max(1e-17,mass.sum()*1e-12) or first>1e-13:rejections['mass_moment']=rejections.get('mass_moment',0)+1;continue
        used.update(rows);proposals.append(r)
    record=dict(witness_capture=digest(accepted),held_out=digest(held),reference=digest(vol/'mesh.node'),training_loaded=digest(loaded),mesh=digest(vol/'mesh.ele'),eligible_nodes=len(nodes),tested_parent_cells=tested,geometric_trials=geometrical,rejected=rejections,passing_sites=len(proposals),repaired_at_capture=sum(len(r['resolved_cells'])for r in proposals),selected=proposals)
    out.write_text(json.dumps(record,indent=2)+'\n');print({k:v for k,v in record.items()if k!='selected'})
if __name__=='__main__':build(*(Path(p).resolve()for p in sys.argv[1:]))
