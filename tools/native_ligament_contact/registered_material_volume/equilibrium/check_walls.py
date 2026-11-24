import argparse
from pathlib import Path
import numpy as np
from audit_material_walls import run

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('volume','fit','checker'):p.add_argument(n,type=Path)
    a=p.parse_args()
    x=np.loadtxt(a.fit/'current.node',skiprows=1,usecols=(1,2,3))
    w=np.load(a.volume/'material-walls.npz')
    run(x,[w[k] for k in 'OIDC'],a.fit/'wall-crossings.json',a.checker)
