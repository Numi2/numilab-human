"""Test the cubic range and all three endpoint derivatives used for a flagged cell."""
import json,sys
from pathlib import Path
import numpy as np

def cof(a):
    return np.stack([np.cross(a[..., :,1],a[..., :,2]),np.cross(a[..., :,2],a[..., :,0]),np.cross(a[..., :,0],a[..., :,1])],axis=-1)
def coefficients(a,b):
    x=np.linalg.det(a);y=np.linalg.det(b)
    return np.column_stack((x,x+np.einsum('ijk,ijk->i',cof(a),b-a)/3,y-np.einsum('ijk,ijk->i',cof(b),b-a)/3,y))
def run(out):
    rng=np.random.default_rng(72591);a=rng.normal(size=(2048,3,3));b=rng.normal(size=a.shape)
    # Both endpoints inside the material bounds while the middle is inverted or singular.
    a[0]=np.eye(3);b[0]=np.diag([-1.,-1.,1.])
    d=rng.normal(size=a.shape);c=coefficients(a,b);max_error=0.
    for u in np.linspace(0,1,65):
        expected=c@np.array([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3])
        actual=np.linalg.det(a+u*(b-a));max_error=max(max_error,np.max(abs(actual-expected)))
        if np.any(actual<c.min(1)-5e-13) or np.any(actual>c.max(1)+5e-13):raise ValueError('Envelope failure')
    if c[0,1:3].min()>=.35 or c[0,0]!=1 or c[0,3]!=1:raise ValueError('Interior failure undetected')
    h=1e-5;fd=(coefficients(a,b+h*d)-coefficients(a,b-h*d))/(2*h)
    grad=[cof(a)/3,(cof(a+b)-cof(a)-cof(b))/3,cof(b)]
    error=[np.max(abs(fd[:,i+1]-np.einsum('ijk,ijk->i',g,d))) for i,g in enumerate(grad)]
    if max(error)>1e-8 or max_error>5e-13:raise ValueError('Derivative or determinant mismatch')
    (out/'path-math.json').write_text(json.dumps(dict(matrices=len(a),fractions=65,seed=72591,maximum_polynomial_error=max_error,maximum_derivative_errors=error,endpoint_counterexample=c[0].tolist()),indent=2)+'\n')
if __name__=='__main__':run(Path(sys.argv[1]))
