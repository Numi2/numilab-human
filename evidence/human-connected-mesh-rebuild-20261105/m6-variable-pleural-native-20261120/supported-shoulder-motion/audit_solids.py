"""Generalized winding of fixed shoulder points in the opposing registered bone."""
import json
from pathlib import Path
import sys
import numpy as np
from shoulder_inputs import transforms, load


def winding(tri,p):
    a,b,c=(tri[:,i,:]-p for i in range(3))
    la,lb,lc=(np.linalg.norm(x,axis=1) for x in (a,b,c))
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    return float(np.arctan2(numerator,denominator).sum()/(2*np.pi))


def launch_files():
    return list(map(Path,sys.argv[1:4]))


if __name__=='__main__':
    _,catalog,meshes,anchors,poses=load(*launch_files())
    for row in anchors['candidates']:
        tri=[]; xyz=[]; bodies=[]
        for sid,face,weight in zip(row['stable_ids'],row['triangle_rows'],row['material_barycentric']):
            v,f,*_=meshes[sid];tri.append(v[f]);xyz.append(np.array(weight)@v[f[face]]);bodies.append(catalog[sid]['body'])
        for k in (32,256,512,8000,-1,-2,-3):
            winding_numbers=[]
            for i in (0,1):
                tr,rot=transforms(poses[k],bodies[i],bodies[1-i])
                other=tr+rot.apply(xyz[1-i])
                winding_numbers.append([round(winding(tri[i],other+t*(xyz[i]-other)),7) for t in (0,.5,.9,.99)])
            print(row['side'],k,winding_numbers)
