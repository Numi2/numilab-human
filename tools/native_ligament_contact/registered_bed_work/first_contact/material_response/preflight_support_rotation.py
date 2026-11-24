"""Test a rigid-body support placement against the emitted Float32 tissue mesh and native cell predicate."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from supported_material_admission import geom

def run(job,volume,fit,support,out):
    predicted=json.loads(support.read_bytes());q=predicted['candidate_rigid_support']
    if digest(fit/'current.node')!=predicted['position_sha256']:raise ValueError('Unmatched body')
    def read(path):return np.loadtxt(path,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    x=read(volume/'mesh.node');y=read(fit/'current.node')
    t=np.ascontiguousarray(np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1)
    z=np.load(job/'registered-boundary/registered-boundary.npz');outer=z['O_node_joint']
    n=np.asarray(q['vertical_after_rotation']);axis=np.cross(n,[0,0,1.])
    R=Rotation.from_rotvec(axis*np.arctan2(np.linalg.norm(axis),n[2])/np.linalg.norm(axis))
    c=np.asarray(predicted['gravity']['center_m'])
    delta=predicted['bed_z_m']+(y[outer,2].min()-predicted['bed_z_m'])-((R.apply(y[outer]-c)+c)[:,2].min())
    shift=c-R.apply(c)+np.array([0,0,delta]);points=lambda v:np.ascontiguousarray((R.apply(v)+shift).astype('f4').astype(float))
    X,Y=points(x),points(y);gain=R.apply(y[q['support_nodes']])+shift
    clearance=Y[outer,2]-predicted['bed_z_m']
    if clearance.min()<0 or max(clearance[np.flatnonzero(np.isin(outer,q['support_nodes']))])>=predicted['activation_width_m']:
        raise ValueError('Rigid pose does not engage all measured witnesses')
    txt=(job/'storage/interval-checker-locator.txt').read_text().split()
    if digest(Path(txt[1]))!=txt[0]:raise ValueError('Unknown determinant checker')
    native=ctypes.CDLL(txt[1]).scanCellIntervals;F=np.ctypeslib.ndpointer(dtype='f8',flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype='i4',flags='C_CONTIGUOUS')
    native.argtypes=[F,F,ctypes.c_size_t,I,ctypes.c_size_t,F]
    val=np.empty((len(t),3));base=np.empty_like(val)
    if native(x,y,len(x),t,len(t),base):raise ValueError('Control failed native predicate')
    # The native scanner returns on its first inadmissible reference cell. Retain
    # those indices and measure the permissible remainder in the same ordering.
    vol=np.concatenate([geom(X,t[i:i+32768])[0] for i in range(0,len(t),32768)])
    valid=np.flatnonzero(vol>1e-18)
    kept=np.ascontiguousarray(t[valid]);values=np.empty((len(kept),3))
    if native(X,Y,len(X),kept,len(kept),values):raise ValueError('Filtered native interval scan failed')
    margin=np.minimum(values[:,0]-values[:,1]-.35,2.5-values[:,0]-values[:,1])
    loss=[];min_vol=float('inf');min_quality=float('inf')
    for i in range(0,len(t),32768):
        a,b=geom(X,t[i:i+32768]);v,w=geom(Y,t[i:i+32768])
        min_vol=min(min_vol,*[float(z.min())for z in (a,v)]);min_quality=min(min_quality,*[float(z.min())for z in (b,w)])
        loss.extend((i+np.flatnonzero((a<=1e-18)|(v<=1e-18)|(b<1e-5)|(w<1e-5))).tolist())
    result=dict(native_checker_sha256=txt[0],parent=digest(support),rigid_rotation_world_xyzw=R.as_quat().tolist(),translation_world_m=shift.tolist(),
        activation_nodes=outer[clearance<predicted['activation_width_m']].tolist(),min_gap_m=float(clearance.min()),
        cells=len(t),reference_cells_outside_native_domain=int(len(t)-len(valid)),tested_intervals=len(valid),failed_native_volume_intervals=int(np.count_nonzero(~np.isfinite(margin)|(margin<=0))),
        failed_absolute_geometry_cells=len(loss),first_geometry_witnesses=loss[:16],minimum_cell_volume_m3=min_vol,
        minimum_cell_quality=min_quality,max_volume_ratio_change_after_emission=float(np.max(abs(values[:,0]-base[valid,0]))),
        scope='Diagnostic rigid placement of the complete skin/fat volume, not an equilibrated or installed native scene.')
    # A superposed rigid motion does not alter the material reference. Use the
    # existing native high/low displacement representation on unchanged rest cells.
    placed=np.ascontiguousarray(R.apply(y)+shift)
    preserved=np.empty_like(base)
    if native(x,placed,len(x),t,len(t),preserved):raise ValueError('Immutable reference was rejected')
    pm=np.minimum(preserved[:,0]-preserved[:,1]-.35,2.5-preserved[:,0]-preserved[:,1])
    nbad=np.flatnonzero(~np.isfinite(pm)|(pm<=0))
    absolute=[];max_det_error=0.
    for i in range(0,len(t),32768):
        sl=t[i:i+32768]
        original,_=geom(y,sl);a,b=geom(placed,sl)
        absolute.extend((i+np.flatnonzero((a<=1e-18)|(b<1e-5))).tolist())
        max_det_error=max(max_det_error,float(np.max(abs(a/original-1))))
    gap=placed[outer,2]-predicted['bed_z_m']
    result['immutable_reference_compensated']=dict(reference_sha256=digest(volume/'mesh.node'),
        failed_native_volume_intervals=len(nbad),first_native_witnesses=nbad[:16].tolist(),
        minimum_native_margin=float(pm.min()),max_native_determinant_change=float(np.max(abs(preserved[:,0]-base[:,0]))),
        failed_absolute_geometry_cells=len(absolute),maximum_relative_geometric_volume_change=max_det_error,
        contacted_outer_nodes=outer[gap<predicted['activation_width_m']].tolist(),
        minimum_gap_m=float(gap.min()),source_coordinates_and_masses_modified=False)
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(a).resolve() for a in sys.argv[1:]))
