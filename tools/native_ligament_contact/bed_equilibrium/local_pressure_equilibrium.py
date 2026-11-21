"""Solve force and moment balance using only registered vertical columns under compression."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from material_normal_response import load,make_chart,compare_virtual_work

class ColumnLaw:
    def __init__(self,geom,skin,fat,mode,center):
        self.x=geom['xyz'];self.A=geom['area_m2'];self.H=geom['layer_heights_m']
        self.z=self.x[:,2]
        self.B=np.column_stack((np.ones(len(self.z)),self.x[:,:2]-center))
        self.p=np.r_[0.,np.geomspace(.00001,5e6,8192)]
        self.curves=[]
        for mat in (skin,fat):
            p,c,j,e=make_chart(mat,mode)
            self.curves.append([np.interp(self.p,p,a) for a in (c,j,e)])
        self.missing=geom['missing']
        self.center=center
        self.gate=[skin['minimum_J'],fat['minimum_J']]
    def evaluate(self,q):
        target=self.B@q-self.z
        # The pressure-indentation curve is monotone and uses the same pressure
        # for the two layers. Locate its active segment without a dense table
        # per column and recover the exact tangent of the interpolation.
        left=np.zeros(len(target),dtype=int);right=np.full(len(target),len(self.p)-1)
        s,f=self.curves[0][0],self.curves[1][0];h,t=self.H.T
        for _ in range(14):
            mid=(left+right)//2
            below=h*s[mid]+t*f[mid]<=target
            left=np.where(below,mid,left);right=np.where(below,right,mid)
        left=np.minimum(left,len(self.p)-2);right=left+1
        dlo=h*s[left]+t*f[left];diff=h*(s[right]-s[left])+t*(f[right]-f[left])
        k=(self.p[right]-self.p[left])/diff
        p=np.where(target>0,self.p[left]+(target-dlo)*k,0.)
        force=p*self.A;active=target>0
        tangent=self.B.T@((k*self.A*active)[:,None]*self.B)
        j=np.stack([np.interp(p,self.p,a[1]) for a in self.curves],axis=1)
        sound=bool((p<self.p[-1]).all() and (j>np.array(self.gate)).all())
        return self.B.T@force,tangent,p,j,sound
    def equilibrium(self,weight,initial):
        target=np.array([weight,0.,0.]);scale=np.array([1.,4.,1.])
        def fun(q):return (self.evaluate(q)[0]-target)*scale
        def jac(q):return self.evaluate(q)[1]*scale[:,None]
        fit=least_squares(fun,initial,jac=jac,xtol=1e-13,ftol=1e-13,gtol=1e-10,max_nfev=100,x_scale='jac')
        wrench,_,p,j,valid=self.evaluate(fit.x)
        if np.max(abs(wrench-target))>max(1.e-5,weight*1e-7):raise ValueError(('Unresolved spatial equilibrium',wrench,target))
        ix=int(p.argmax());active=p>0
        outside=int(np.count_nonzero(p>self.p[-1]))
        incomplete=self.missing
        omitted=(fit.x[0]+(incomplete[:,1:3]-self.center)@fit.x[1:]-incomplete[:,3])>0
        result=dict(weight_n=weight,generalized_reaction=wrench.tolist(),maximum_balance_error=float(max(abs(wrench-target))),
                    vertical_approach_m=float(fit.x[0]),gradients=fit.x[1:].tolist(),loaded_samples=int(active.sum()),
                    loaded_area_m2=float(self.A[active].sum()),table_exceeded_samples=outside,witness=ix,
                    omitted_engaged_columns=int(omitted.sum()),omitted_engaged_area_m2=float(incomplete[omitted,4].sum()),
                    valid=valid,qualified_within_sampled_material_and_coverage=valid and not omitted.any())
        # A continuation outside the table locates a failed guess only. Its extrapolated
        # stresses and determinants do not belong to the material model.
        if not outside:
            bad=np.any(j<=np.array(self.gate),axis=1)
            result.update(peak_pressure_pa=float(p[ix]),minimum_J=j.min(0).tolist(),
                          invalid_samples=int(bad.sum()),force_on_invalid_samples_n=float(p[bad]@self.A[bad]))
        return fit.x,result

def run(geofile,matdir,moments,out):
    geom=np.load(geofile);d=json.loads(moments.read_text())
    weight=d.get('weight_n',d.get('total_body_weight_at_9_81_N_kg_N'))
    center=np.array(d.get('center_xy_m',d.get('centre_of_accounted_mass_xy_m')))
    mats=[load(matdir/name) for name in ('human_skin_dermis_frequency_specific_effective.nmatter','human_subcutaneous_adipose_low_strain_effective.nmatter')]
    result=dict(scope='Static normal material-column predictor at the measured registered entrance; transverse boundary conditions are explicit assumptions; no invented skeleton constraint applied to Matter',center_xy_m=center.tolist(),materials=mats,quadrature=geofile.name,results={})
    for mode in ('confined','traction_free'):
        model=ColumnLaw(geom,*mats,mode,center)
        q=np.array([.004,0.,0.]);rows=[];lastgood=None;firstbad=None
        for fraction in [1e-3,.01,.05,.1,.25,.5,.75,1.]:
            q,row=model.equilibrium(weight*fraction,q);rows.append(row)
            if row['valid'] and firstbad is None:lastgood=(fraction,q.copy())
            if not row['valid'] and firstbad is None:firstbad=(fraction,q.copy())
        boundary=None
        if lastgood is not None and firstbad is not None:
            lo,lq=lastgood;hi,hq=firstbad
            for i in range(20):
                mid=(lo+hi)/2;q,row=model.equilibrium(weight*mid,(lq+hq)/2)
                if row['valid']:lo,lq=mid,q
                else:hi,hq=mid,q
            _,terminal=model.equilibrium(weight*lo,lq)
            boundary=dict(last_valid_weight_fraction=lo,first_invalid_weight_fraction=hi,last_valid=terminal)
        ind=rows[-1]['witness'];rows[-1]['peak_outer_inner_deep_face_ids']=geom['ids'][ind].tolist()
        result['results'][mode]=dict(samples=rows,first_material_limit=boundary,energy_virtual_work_relative_errors=[compare_virtual_work(mat,mode) for mat in mats])
    out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:(v['samples'][-1]['loaded_samples'],v['first_material_limit']['last_valid_weight_fraction'],v['samples'][-1]['table_exceeded_samples']) for k,v in result['results'].items()})
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
