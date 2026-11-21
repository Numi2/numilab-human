"""Apply source-derived shoulder-base calibration to the two existing NHRIGID2 joint rows."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(b):return hashlib.sha256(b).hexdigest()

def apply(launch,calibration,out):
    config=json.loads(launch.read_text()); path=Path(config['argv'][1])
    source=path.read_bytes()
    assert sha(source)==config['pins'][str(path)]
    assert source[:8]==b'NHRIGID2'
    version,engine,_,nb,nj,*_=struct.unpack_from('<10I',source,8)
    assert (version,engine,nb,nj)==(1,5,157,156)
    data=bytearray(source)
    rows=json.loads(calibration.read_text()); receipts=[]
    assert [r['side'] for r in rows]==[0,1]
    for r,parent,joint in zip(rows,(84,34),(84,34)):
        trial=r['trials'][-1]
        assert trial['additional_shift_mm']==0 and len(trial['counts'])==254 and max(trial['counts'].values())==0
        off=224+160*nb+joint*144
        parent_id,child,kind,_=struct.unpack_from('<4I',source,off)
        assert (parent_id,child,kind)==(parent,parent+1,0)
        tr=np.asarray(r['parent_translation_base_m']); rot=Rotation.from_quat(r['parent_rotation_xyzw'])
        a=off+80; q=off+112
        prev_a=struct.unpack_from('<3f',source,a);prev_q=struct.unpack_from('<4f',source,q)
        new_a=rot.apply(prev_a)+tr; new_q=(rot*Rotation.from_quat(prev_q)).as_quat()
        struct.pack_into('<3f',data,a,*new_a);struct.pack_into('<4f',data,q,*new_q)
        receipts.append(dict(side=r['side'],joint=joint,parent=parent,child=child,previous_anchor_m=prev_a,
                             candidate_anchor_m=struct.unpack_from('<3f',data,a),previous_quaternion=prev_q,
                             candidate_quaternion=struct.unpack_from('<4f',data,q)))
    touched={224+160*nb+j*144+i for j in (34,84) for i in (*range(80,92),*range(112,128))}
    assert all(i in touched for i,(x,y) in enumerate(zip(source,data)) if x!=y)
    with out.open('xb') as f:f.write(data)
    return dict(source=str(path),source_sha256=sha(source),calibration_sha256=sha(calibration.read_bytes()),
                result_sha256=sha(data),rows=receipts,changed_byte_count=sum(x!=y for x,y in zip(data,source)))
if __name__=='__main__':
    *args,receipt=map(Path,sys.argv[1:]);receipt.write_text(json.dumps(apply(*args),indent=2)+'\n')
