"""Compare changed bone banks with the full captured physical-floor anatomy at time zero."""
import hashlib,json,os,runpy,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import old,digest
from bone_overlap import intersections
def run(scene,loader,zero,bone,out):
    os.environ['NUMI_HUMAN_CAPTURED_TIME_ZERO_RECEIPT']=str(zero)
    sys.argv[1]=str(scene)
    ref=runpy.run_path(str(loader))
    poses=ref['poses'];banks=ref['banks'];banks[ref['a']['SKIN']]=(ref['skin'],ref['skin_faces'])
    cfg=json.loads((scene/'launch.json').read_text());parent=Path(cfg['argv'][3]);old.audit.ag.RIGID=Path(cfg['argv'][1])
    geom=[]
    for path in (parent,bone):
        old.audit.ag.BONES=path;_,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
        geom.append(old.load_local_meshes(raw,cat,vo,io))
    results=[];prepared={k:(v.min(0),v.max(0)) for k,(v,f) in banks.items()}
    for b in (8,9,10,11):
        p=poses[cat[b]['body']];rr=R.from_quat(p['quaternion_xyzw']);tr=p['position_m']
        v0,v1=[(rr.apply(ms[b][0])+tr).astype(np.float32) for ms in geom]
        v,f=banks[(51004,b)]
        discrepancy=max(cKDTree(v).query(v0)[0].max(),cKDTree(v0).query(v)[0].max())
        assert discrepancy<2e-6,(b,discrepancy)
        children=[];fc=geom[1][b][1];assert np.array_equal(fc,geom[0][b][1])
        for k,(x,xf) in banks.items():
            if k[0]==51004:continue
            counts=[];ex=[]
            for q in (v0,v1):
                lo,hi=prepared[k]
                if np.any(q.min(0)>hi) or np.any(q.max(0)<lo):pairs=[]
                else:pairs=intersections(q,fc,x,xf)
                counts.append(len(pairs));ex.append(pairs[:1])
            if any(counts) or k==ref['a']['SKIN']:children.append(dict(semantic=k[0],stable=k[1],parent=counts[0],candidate=counts[1],examples=ex))
        results.append(dict(bone=b,baseline_decoder_max_error_m=float(discrepancy),rows=children))
        print(b,sum(r['parent'] for r in children),sum(r['candidate'] for r in children),next(r for r in children if r['semantic']==51007 and r['stable']==1),flush=True)
        out.write_text(json.dumps(dict(bones_sha256=digest(bone),pose_receipt_sha256=digest(zero),base_geometry_sha256=ref['zero']['base_pack_file_sha256'],results=results),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
