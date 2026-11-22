"""Reconstruct the anterior material cavity as a common vertical shift of all three layers."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,maximum_filter,map_coordinates,distance_transform_edt
from scipy.spatial import cKDTree
from emit_skin_source_registration import sha

def heights(x,f,axes):
    """Uppermost actual projected face at every grid point with deterministic face selection."""
    gx,gy=axes;out=np.full((len(gx),len(gy)),-np.inf);step=gx[1]-gx[0]
    tri=x[f].astype('f8');lo=tri.min(1);hi=tri.max(1)
    i0=np.ceil((lo[:,:2]-[gx[0],gy[0]])/step-1e-9).astype(int)
    i1=np.floor((hi[:,:2]-[gx[0],gy[0]])/step+1e-9).astype(int)
    i0=np.maximum(i0,0);i1=np.minimum(i1,[len(gx)-1,len(gy)-1])
    for n in np.flatnonzero(np.all(i1>=i0,axis=1)):
        a,b,c=tri[n];e=b-a;g=c-a;det=e[0]*g[1]-g[0]*e[1]
        if abs(det)<1e-20:continue
        i,j=i0[n];k,l=i1[n]+1;px=gx[i:k,None]-a[0];py=gy[None,j:l]-a[1]
        v=(px*g[1]-py*g[0])/det;w=(py*e[0]-px*e[1])/det
        ok=(v>=-1e-9)&(w>=-1e-9)&(v+w<=1+1e-9)
        z=np.where(ok,a[2]+v*e[2]+w*g[2],-np.inf);np.maximum(out[i:k,j:l],z,out=out[i:k,j:l])
    return out

def smooth(t):
    v=np.clip(t,0.,1.);return v*v*(3.-2.*v)
def main():
    p=argparse.ArgumentParser()
    for k in ('input','subdiv','out'):p.add_argument(k,type=Path)
    p.add_argument('--spacing',type=float,default=.003);p.add_argument('--monotone',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    s=np.load(args.input);z=np.load(args.subdiv);h=args.spacing
    axes=(np.arange(-.245,.194,h),np.arange(-.926,-.296,h))
    x=s['x'];o,i,d=[heights(x,s[k],axes) for k in 'OID'];q=heights(s['source_x'],s['source_f'],axes)
    mus=np.max([heights(s['x'+str(k)],s['f'+str(k)],axes) for k in (67,68,73,74)],axis=0)
    # A matched sheet includes all three ordered depths. Never use rays through an arm or a void to set a torso thickness.
    good=np.isfinite(mus)&np.isfinite(q)&np.isfinite(o)&np.isfinite(i)&np.isfinite(d)
    with np.errstate(invalid='ignore'):
        good&=(o>i)&(i>d)&(o-d<.047)&(abs(q-o)<.027)&(q>mus-1e-5)
        correction=np.where(good,np.maximum.reduce([np.zeros_like(q),q-o,mus-d+.00025]),0.)
    # Retain the measured obstacle height under a smooth, local clearance envelope.
    envelope=gaussian_filter(maximum_filter(correction,size=7),sigma=1.5,mode='constant')
    field=np.maximum(correction,envelope)
    if args.monotone:
        # Pin the complete material cut. Cover the projected boxes of all cut
        # facets that reach the vertical transition, including interpolation cells.
        blocked=np.zeros_like(field,dtype=bool);origin=np.array([v[0] for v in axes])
        cut=x[np.concatenate([s['C15'],s['C16']])]
        for tri in cut[cut[:,:,2].max(1)>.078]:
            imin=np.maximum(0,np.floor((tri[:,:2].min(0)-origin)/h).astype(int)-1)
            imax=np.minimum(np.array(field.shape),np.ceil((tri[:,:2].max(0)-origin)/h).astype(int)+2)
            if np.all(imax>imin):blocked[imin[0]:imax[0],imin[1]:imax[1]]=True
        field*=smooth(distance_transform_edt(~blocked)*h/.016)
    base=z['unmoved' if 'unmoved' in z else 'x'].astype('f8');coords=np.column_stack([(base[:,k]-axes[k][0])/h for k in (0,1)]).T
    sampled=map_coordinates(field,coords,order=1,mode='constant')
    dc=cKDTree(x[s['fix']]).query(base)[0]
    alpha=smooth((base[:,2]-.078)/.042)
    if not args.monotone:alpha*=smooth(dc/.021)
    shift=sampled*alpha;shift[s['fix']]=0
    y=base.copy();y[:,2]+=shift;y=y.astype('f4')
    data={k:z[k] for k in ['O','I','D','C15','C16']}
    for k in 'OID':data['parent_'+k]=z['parent_'+k] if 'parent_'+k in z else np.arange(len(z[k]),dtype='i4')
    data['original_nodes']=z['original_nodes'] if 'original_nodes' in z else z['nodes']
    data.update(x=y,unmoved=base.astype('f4'),chart=field,origin=np.array([a[0] for a in axes]),spacing=h,monotone=args.monotone);np.savez_compressed(args.out/'proposal.npz',**data)
    assert np.array_equal(y[s['fix']],x[s['fix']])
    r=dict(scope='three material boundaries with pinned connected cell cut, clearance constrained by anatomical muscle facets and authored tissue depth',input_sha256=sha(args.input),subdivision_sha256=sha(args.subdiv),proposal_sha256=sha(args.out/'proposal.npz'),grid_spacing_m=h,rays=int(good.sum()),rays_requiring_depth=int(np.sum(good&(mus>d))),max_movement_mm=float(1000*shift.max()),p99_nonzero_movement_mm=float(1000*np.quantile(shift[shift>0],.99)),cut_nodes_fixed=len(s['fix']))
    (args.out/'metrics.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r,flush=True)
if __name__=='__main__':main()
