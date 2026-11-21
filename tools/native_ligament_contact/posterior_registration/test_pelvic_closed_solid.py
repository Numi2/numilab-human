"""Check separation, containment, and invalid surfaces with native-sign predicates."""
import json
import sys
from pathlib import Path
import numpy as np
from pelvic_closed_solid import prepare,compare

def exercise(predicate):
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    xyz=np.array([[0,0,0],[4,0,0],[0,4,0],[0,0,4]],dtype='f4').astype(float)
    shell=prepare(xyz,faces,predicate)
    inside=prepare(xyz*.125+.25,faces,predicate)
    moved=prepare(xyz+[6.,6.,6.],faces,predicate)
    missing=prepare(xyz,faces[1:],predicate)
    flipped=faces.copy();flipped[0]=flipped[0,::-1]
    reverse=prepare(xyz,flipped,predicate)
    if not compare(shell,moved)['closed_and_exterior']:raise ValueError('Separated solids')
    for left,right,key in ((shell,inside,'cortex_in_material'),(inside,shell,'material_in_cortex')):
        result=compare(left,right)
        if result['closed_and_exterior'] or result['component_tests'][key][0]['location']!='inside':
            raise ValueError('Contained component missed')
    if missing[3]['open_or_nonmanifold_edges']!=3 or reverse[3]['inconsistent_edges']!=3:
        raise ValueError('Invalid boundary accepted')
    print(json.dumps({'separated':True,'both_containment_directions':True,'missing_face':True,'orientation_defect':True}))
if __name__=='__main__':exercise(Path(sys.argv[1]))
