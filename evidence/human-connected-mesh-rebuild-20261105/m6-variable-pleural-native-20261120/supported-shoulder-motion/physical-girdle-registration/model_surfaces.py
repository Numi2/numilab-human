"""Extract shoulder surfaces and joint axes from the existing composed MyoSim source model."""
import json
from pathlib import Path
import sys
import hashlib
import mujoco
from myo_sim.build.compose import build_model
import numpy as np


def extract(archive, out):
    model = build_model("myofullbody")
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    inventory = []
    arrays = {}
    for side in ("l", "r"):
        for name in ("scapula","clavicle","humerus","radius","ulna"):
            full = f"{name}_{side}"
            b = model.body(full).id
            ids = [g for g in range(model.ngeom) if model.geom_bodyid[g]==b and
                   model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH]
            assert len(ids)==1,(full,ids)
            g=ids[0]; m=int(model.geom_dataid[g])
            start=int(model.mesh_vertadr[m]); count=int(model.mesh_vertnum[m])
            fs=int(model.mesh_faceadr[m]); nf=int(model.mesh_facenum[m])
            v=model.mesh_vert[start:start+count].astype(float)
            f=model.mesh_face[fs:fs+nf].copy()
            assert f.min()>=0 and f.max()<count
            world=v@state.geom_xmat[g].reshape(3,3).T+state.geom_xpos[g]
            ir=state.ximat[b].reshape(3,3)
            local=(world-state.xipos[b])@ir
            arrays[full+"_world"]=world;arrays[full+"_local"]=local;arrays[full+"_faces"]=f
            inventory.append(dict(body=full,source_body=b,mesh_id=m,geom_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),
                vertices=count,faces=nf,extents_m=np.ptp(world,axis=0).tolist(),sha256_world_float64=hashlib.sha256(world.tobytes()).hexdigest()))
        for name in ("shoulder_rot","elbow_flexion","shoulder_elv"):
            j=model.joint(f"{name}_{side}").id
            arrays[f"{name}_{side}_center"]=state.xanchor[j].copy()
            arrays[f"{name}_{side}_axis"]=state.xaxis[j].copy()
    np.savez_compressed(out/"source-bones.npz",**arrays)
    identity=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
    assert identity=="280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
    (out/"model-source.json").write_text(json.dumps(dict(source_archive_sha256=identity,mujoco=mujoco.__version__,bones=inventory),indent=2)+'\n')
if __name__=="__main__":extract(*map(Path,sys.argv[1:]))
