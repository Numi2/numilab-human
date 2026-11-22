"""Bind the dense registered outer nodes to a condensed-bed launch from the same skeleton."""
import json,sys
from pathlib import Path
from emit_skin_source_registration import sha
base, full, out=map(Path,sys.argv[1:])
cfg=json.loads(base.read_text()); material=json.loads(full.read_text())
assert sha(Path(cfg['argv'][1]))==sha(Path(material['argv'][1]))
assert cfg['environment']['NUMI_HUMAN_LOCAL_BED_LAYERS']=='1'
assert material['environment']['NUMI_HUMAN_FEM_SKIN_SUPPORT']=='1'
assert cfg['environment'].get('NUMI_HUMAN_FEM_SKIN_SUPPORT','0')=='0'
prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
for name in ('SURFACE_O_NODES_CSV','SURFACE_O_NODES_SHA256','SURFACE_O_FACES_CSV',
             'SURFACE_O_FACES_SHA256','SURFACE_O_STABLE_IDENTIFIER','ROLE_O_SOURCE_ASSET',
             'ROLE_O_SOURCE_ASSET_SHA256','ROLE_O_SOURCE_DOMAIN','ROLE_O_SOURCE_VERTEX_COUNT',
             'ROLE_O_SOURCE_FACE_COUNT'):
    cfg['environment'][prefix+name]=material['environment'][prefix+name]
for name,pin in (('SURFACE_O_NODES_CSV','SURFACE_O_NODES_SHA256'),
                 ('SURFACE_O_FACES_CSV','SURFACE_O_FACES_SHA256'),('ROLE_O_SOURCE_ASSET','ROLE_O_SOURCE_ASSET_SHA256')):
    path=cfg['environment'][prefix+name]
    assert sha(Path(path))==cfg['environment'][prefix+pin]
    cfg['pins'][path]=cfg['environment'][prefix+pin]
cfg['environment']['NUMI_HUMAN_LOCAL_BED_REGISTERED_OUTER']='1'
out.mkdir(exist_ok=False)
(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
(out/'run_supported.py').hardlink_to(base.parent/'run_supported.py')
