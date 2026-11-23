"""Check the emitted FEM reference O surface against the *same* opening anatomy banks."""
import json,hashlib,subprocess,runpy,sys
from pathlib import Path
import numpy as np
from material_context import sha
def check(case,opening,source,reader,bank):
    identity=json.loads((source/'support-wrench.json').read_text())
    rec=json.loads((case/'terminal.receipt.json').read_text());base=json.loads((opening/'terminal.receipt.json').read_text())
    if sha(opening/'terminal.receipt.json')!=identity['pose'] or rec['base_pack_file_sha256']!=base['base_pack_file_sha256']:raise ValueError('mixed registrations')
    if sha(source/'opening-outer.npz')!=identity['mesh']:raise ValueError('different physical surface')
    raw=subprocess.check_output(['zstd','-qdc',case/'initial.mrvpack.zst'])
    if hashlib.sha256(raw).hexdigest()!=rec['base_pack_file_sha256']:raise ValueError('different source pack')
    poses={r['body_index']:r for r in base['initial_anatomical_registration']['body_poses']}
    if sorted(poses)!=list(range(rec['accepted_body_pose_count'])):raise ValueError('incomplete common skeleton')
    sys.path.insert(0,str(reader.parent/'src'))
    module=runpy.run_path(str(reader));scan=runpy.run_path(str(bank))['scan_banks']
    four=np.load(source/'opening-outer.npz');v=np.ascontiguousarray(four['vertices'],dtype=np.float32);f=np.ascontiguousarray(four['triangles'],dtype=np.int32)
    out=source/'outer-envelope';out.mkdir(exist_ok=True)
    obs=dict(pack_sha256=rec['base_pack_file_sha256'],physical_outer_node_sha256=identity['source_boundary'],mesh_reference_sha256=identity['mesh'],
             skin_vertices=len(v),skin_faces=len(f),scope='Registered stress-free FEM outer reference evaluated in the same opening skeleton. Rigid donor registration does not simulate soft tissue loading.')
    scan(module,raw,poses,v,f,obs,out,'physical-outer.json',check_skin=True)
    subprocess.run([sys.executable,bank.parent/'classify_registered_envelope.py',out/'physical-outer.json',case/'stdout.log',out/'physical-banks.json','--window-receipt',case/'resting-ocular-sclera-window.json'],check=True)
if __name__=='__main__':check(*(Path(x).resolve() for x in sys.argv[1:]))
