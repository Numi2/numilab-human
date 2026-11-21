"""Regression for ordinary axial fibers and rejected-step generalized load."""
import json
import struct
import subprocess
import tempfile
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp)
    row=struct.pack('<4I16f',0,1,1,0,0,.04,0,0,0,.04,0,0,1.e5,1.,1.e5,1.05,.1,.0001,1.,0)
    (d/'rows').write_bytes(row+bytes(768))
    gap=np.array([.09,.1,.101,.104])
    poses=np.zeros((len(gap),2,8),'<f4');poses[:,:,7]=1;poses[:,1,0]=gap;poses.tofile(d/'poses')
    jac=np.zeros((len(gap),2,4,3,12),'<f4')
    for body in range(2):
        jac[:,body,:,:,6*body:6*body+3]=np.eye(3)
        for i,k in enumerate(np.eye(3)):jac[:,body,i+1,:,6*body+3:6*body+6]=np.cross(np.eye(3),k).T
    jac.tofile(d/'jac')
    subprocess.run([root/'audit_gpu',root/'build/NumiMatter.metallib',d/'rows',d/'poses',d/'audit',d/'jac',d/'force'],check=True)
    a=np.fromfile(d/'audit','<f4').reshape(-1,12);f=np.fromfile(d/'force','<f4').reshape(-1,12)
    t=10*np.expm1(np.maximum(0,gap/.1-1))
    np.testing.assert_allclose(a[:,7],t,atol=1.e-6)
    np.testing.assert_allclose(a[:,8],gap/.1,atol=1.e-6)
    expected=t[:,None]*np.array([1,0,0,0,0,-.04,-1,0,0,0,0,.04])[None,:]
    expected[-1]=0
    np.testing.assert_allclose(f,expected,atol=1.e-6)
    print('Straight fiber slack, tension, torque and rejected-step load passed in production Metal kernels.')
