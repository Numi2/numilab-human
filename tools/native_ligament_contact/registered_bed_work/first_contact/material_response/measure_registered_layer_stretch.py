"""Measure in-plane deformation of the actual emitted triangular skin and fat interfaces."""
from pathlib import Path
import json,sys
import numpy as np
from compile_connected_joint import digest
def run(job,material,out):
    root=job/'interval-certified-scene/material';read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    x,y=read(root/'mesh.node'),read(material/'current.node');z=np.load(job/'registered-boundary/registered-boundary.npz');record={}
    for k in 'OID':
        f=z[k+'_node_joint'][z[k+'_face_nodes']];st=[];area=[]
        for i in range(0,len(f),16384):
            v=f[i:i+16384];a=x[v[:,1:]]-x[v[:,:1]];b=y[v[:,1:]]-y[v[:,:1]]
            q,r=np.linalg.qr(a.transpose(0,2,1),mode='reduced')
            sv=np.linalg.svd(np.linalg.solve(r.transpose(0,2,1),b),compute_uv=False)
            st.append(sv);area.append(.5*np.linalg.norm(np.cross(a[:,0],a[:,1]),axis=1))
        s=np.vstack(st);ar=np.concatenate(area);bad=(s[:,0]>2)|(s[:,1]<.5);worst=np.argmax(np.maximum(s[:,0],1/s[:,1]))
        record[k]=dict(triangles=len(f),severe=int(bad.sum()),reference_area_fraction=float(ar[bad].sum()/ar.sum()),max_stretch=float(s[:,0].max()),min_stretch=float(s[:,1].min()),worst_face=int(worst),worst_source_nodes=f[worst].tolist())
    out.write_text(json.dumps(dict(material=digest(root/'mesh.node'),loaded=digest(material/'current.node'),layers=record),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
