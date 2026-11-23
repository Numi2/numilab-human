"""Measure registration and material capacity over the whole positive-area bed footprint."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
from probe_registered_layers import measure
from inspect_bed_load_paths import intersections
from material_context import sha
from scipy.optimize import brentq
from fit_bed_section_curve import stress

def compare(case,paths,opening,lib):
    j=json.loads((opening/'launch.json').read_text());all_columns=opening/'registered-bed-columns.csv'
    columns=np.genfromtxt(all_columns,delimiter=',',names=True)
    columns=columns[columns['area_m2'].astype('f4')>0]
    n=len(columns);xy=np.ascontiguousarray(np.c_[columns['x_m'],columns['y_m']]);z=columns['z_m'];area=columns['area_m2']
    proof=json.loads((opening/'terminal-bed-certificate.json').read_text())
    if n!=proof['checked_columns'] or sha(all_columns)!=proof['columns']:raise ValueError('Mismatched support quadrature')
    fn=ctypes.CDLL(str(lib.resolve())).bed_boundary_columns
    d=np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
    fn.argtypes=[d,ctypes.c_int,i,ctypes.c_int,d,ctypes.c_int,i,d,d];fn.restype=ctypes.c_int
    table={};checks={};refs=json.loads(paths.read_text())['witnesses']
    ids=np.unique(np.r_[np.linspace(0,n-1,15,dtype=int),[np.flatnonzero(columns['vertex']==int(s))[0] for s in refs]])
    def sample(k,vertices,faces):
        v=np.ascontiguousarray(vertices,dtype=float);f=np.ascontiguousarray(faces,dtype='i4')
        hit=np.empty(n,'i4');h=np.empty(n);b=np.empty((n,3))
        if fn(v,len(v),f,len(f),xy,n,hit,h,b):raise ValueError('Boundary sampling failed')
        present=hit>=0
        vv=np.einsum('ij,ijk->ik',b[present],v[f[hit[present]]])
        er=float(abs(vv-np.c_[xy[present],h[present]]).max())
        if er>1e-9:raise ValueError('Invalid source face ancestry')
        for idx in ids:
            hits=intersections(v,f,np.r_[xy[idx],0.])
            if (not hits)!=bool(not present[idx]) or (hits and abs(hits[0][0]-h[idx])>1e-10):
                raise ValueError('Independent intersection disagrees')
        table[k]=h;checks[k]=dict(numerical_witnesses=len(ids),max_reconstruction_error_m=er,mapped_columns=int(present.sum()),missing_area_m2=float(area[~present].sum()))
    model=measure(case,paths,opening,sample)
    O,I,D=[table[k] for k in 'OID'];present=np.isfinite(O);ordered=present&np.isfinite(I)&np.isfinite(D)&(O<I)&(I<D)
    curve,=[Path(k) for k in j['pins'] if k.endswith('/curves.json')]
    if sha(curve)!=j['pins'][str(curve)]:raise ValueError('Material binding changed')
    materials=json.loads(curve.read_text())['material']['materials']
    coeff=[(m['mu_pa'],m['log_volume_coefficient_pa']) for m in materials]
    cap=min(stress(2/3,*x) for x in coeff);ss,sf=[brentq(lambda y:stress(y,*x)-cap,2/3,1.) for x in coeff]
    depth=(1-ss)*columns['skin_m']+(1-sf)*columns['fat_m']
    gap=O-z;bad=present&(abs(gap)>depth)
    quantiles=[0,.1,.5,.9,.99,1.]
    def quant(a,mask):
        vals=a[mask];ix=np.argsort(vals);cw=np.cumsum(area[mask][ix])
        return np.interp(quantiles,cw/cw[-1],vals[ix]).tolist()
    skin_delta=np.subtract(I,O,where=ordered,out=np.zeros(n))-columns['skin_m']
    fat_delta=np.subtract(D,I,where=ordered,out=np.zeros(n))-columns['fat_m']
    return dict(columns=sha(all_columns),native_opening=sha(opening/'terminal.receipt.json'),dense_material_reference=model['physical_reference'],registered_donor_errors=model['source_frame_evaluation'],positive_columns=n,area_m2=float(area.sum()),layers=checks,all_layers_ordered_columns=int(ordered.sum()),all_layers_ordered_area_m2=float(area[ordered].sum()),quantiles=quantiles,outer_height_difference_m_area_weighted=quant(gap,present),columns_beyond_local_material_allowance=int(bad.sum()),area_beyond_local_material_allowance_m2=float(area[bad].sum()),area_fraction_beyond_local_allowance=float(area[bad].sum()/area.sum()),skin_stock_difference_m_area_weighted=quant(skin_delta,ordered),fat_stock_difference_m_area_weighted=quant(fat_delta,ordered),scope='Registered unloaded O/I/D boundaries versus support points in the identical articulated opening frame. Lowest ray intersections measure projected heights. Missing or unordered layers have no qualified transfer. Bounds are geometric; this test does not measure tissue forces.')
if __name__=='__main__':print(json.dumps(compare(*(Path(s) for s in sys.argv[1:])),indent=2,allow_nan=False))
