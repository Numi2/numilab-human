"""Read the registered physical tissue, material ownership, mass and fixed bed."""
import hashlib
import json
from pathlib import Path
import re
import struct
import numpy as np

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def native_plane(path):
    b=Path(path).read_bytes()
    if struct.unpack_from('<8s4I',b)[:3]!=(b'NHCNT1\0\0',1,157):
        raise ValueError('Unmatched native support and reference adult')
    p,q=struct.unpack_from('<3f',b,56),struct.unpack_from('<3f',b,68)
    if q!=(0.,0.,1.):
        raise ValueError('Expected upward native support')
    return float(p[2])

def mass_ledger(env,t,role,n):
    receipt=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_ALLOCATION_SKIN_RECEIPT_JSON'])
    source=receipt.parents[2]/'fine-mass-routed/current-density-closed-flow.npz'
    cavity=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    digest=sha(source)
    if digest!=read(source.parent/'mass-transfer.json')['flow_sha256'] or sha(cavity)!=env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:
        raise ValueError('Unmatched native material mass')
    with np.load(source) as prior,np.load(cavity) as patch:
        element=np.r_[prior['element_mass_kg'][patch['retained_input_cell_rows']],patch['replacement_mass_kg']]
    if len(element)!=len(t) or not np.all(np.isfinite(element)&(element>0)):
        raise ValueError('Invalid element mass')
    m=np.bincount(t.ravel(),np.repeat(element/4,4),minlength=n)
    if m.min()<=0 or abs(m.sum()-element.sum())>1.e-10:
        raise ValueError('Invalid nodal mass')
    return m,{str(k):float(element[role==k].sum()) for k in (15,16)},digest

def reference(parent):
    from fit_posterior_equilibrium import materials
    case=parent/'native-terminal-16'
    source=parent/'native-input.npz'
    if sha(source)!=read(parent/'harmonic/result.json')['source']:
        raise ValueError('Unmatched material volume')
    cfg=read(case/'launch.json')
    owner=Path(cfg.get('compiled_material_owner',Path(cfg['argv'][0]).parent))
    p=case/'physical-reference.node'
    if sha(p)!=read(case/'accepted-layers.json')['inputs']['NATIVE_REFERENCE_NODE_SHA256']:
        raise ValueError('Unexpected native reference node identity')
    with np.load(source) as z:
        x,t,role,outer=z['x'].astype('f8'),z['t'],z['role'],z['node_O']
    nodes=np.loadtxt(p,skiprows=1)
    if not np.array_equal(nodes[:,1:],x) or not np.array_equal(nodes[:,0],np.arange(1,len(x)+1)):
        raise ValueError('Reference node ordering differs')
    if not np.array_equal(np.unique(role),[15,16]) or t.shape!=(len(role),4) or t.min()<0 or t.max()>=len(x):
        raise ValueError('Invalid skin/fat material topology')
    text=(owner/'NumiHumanLoadedSkinPoseData.inc').read_text().split('constexpr Anchor anchors[]={\n',1)[1].split('};',1)[0]
    anchors=np.array([[int(s) for s in row] for row in re.findall(r'\{(\d+)u,(\d+)u,(\d+)u,\{',text)])
    if anchors.shape!=(14,3) or len(np.unique(anchors[:,1]))!=14 or not np.all((anchors[:,1]>=0)&(anchors[:,1]<len(x))):
        raise ValueError('Unmatched material attachments')
    props,pins=materials(owner)
    if not np.all(np.isfinite(props)) or np.any(props[:,0]<=0) or np.any(props[:,1]+2*props[:,0]/3<0):
        raise ValueError('Reference tangent must be positive semidefinite')
    return x,t,role,outer,anchors[:,1],props,pins
