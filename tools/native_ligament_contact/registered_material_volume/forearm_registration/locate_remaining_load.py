"""Test observer parity and locate terminal force residuals after the connected forearm registration."""
from pathlib import Path
import json,re,sys
import numpy as np
from compile_connected_joint import digest
def run(control,observed,volume,out):
    if digest(control/'native-traces/resting-coupled.csv')!=digest(observed/'native-traces/resting-coupled.csv'):raise ValueError('Different accepted physiology')
    a,b=[json.loads((p/'failure.json').read_bytes()) for p in (control,observed)]
    if a['segment_start']!=b['segment_start'] or not np.allclose(a['matter_diagnostics'][:3],b['matter_diagnostics'][:3],rtol=5e-5,atol=1e-12):raise ValueError('Different failure')
    text=(observed/'stderr.log').read_text()
    records=[]
    for match in re.finditer(r'certificate_row_measure root=1 span=(\d+) offset=(\d+) count=(\d+) mechanics2=(\S+) pressure2=(\S+) peak2=(\S+) local=(\d+)',text):
        si,off,n,m,p,pk,i=match.groups();records.append(dict(span=int(si),offset=int(off),n=int(n),mechanical_squared=float(m),pressure_squared=float(p),peak_squared=float(pk),row=int(i)))
    if not records or len({r['span'] for r in records})!=len(records):raise ValueError('Ambiguous native partition')
    norm=sum(r['mechanical_squared']+r['pressure_squared'] for r in records)**.5
    error=abs(norm-b['matter_diagnostics'][0])/b['matter_diagnostics'][0]
    if error>1e-5:raise ValueError('Partition does not reproduce the actual physical residual')
    peak=max(records,key=lambda r:r['peak_squared']);skin=next(r for r in records if r['span']==15)
    idx=skin['row'];ts=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1;cells=np.flatnonzero((ts==idx).any(1));lab=np.loadtxt(volume/'current-child.labels.csv',skiprows=1,delimiter=',',dtype='i8')[cells]
    r=dict(native_terminal=norm,relative_attribution_error=error,peak_object=peak['span'],skin_fat_peak_node_zero_based=idx,skin_fat_fraction=skin['mechanical_squared']/norm**2,peak_fraction=peak['peak_squared']/norm**2,incident_sources=np.unique(lab[:,1:],axis=0).tolist(),incident_cells=cells.tolist(),trace=digest(observed/'native-traces/resting-coupled.csv'),observer=digest(observed/'launch.json'),spans=records)
    out.write_text(json.dumps(r,indent=2)+'\n');print({k:r[k]for k in ('peak_object','skin_fat_peak_node_zero_based','skin_fat_fraction','peak_fraction','incident_sources')})
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
