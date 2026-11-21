"""Reconstruct the rejected hand registration and verify its retained geometry and mass witnesses on Numi."""
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from compare_hand_interfaces import run as compare
from check_distal_paths import run as paths
from test_changed_surfaces import run as fixtures
from check_hand_candidate import sha

def check_equal(a,b):
    if json.loads(a.read_text())!=json.loads(b.read_text()):
        raise ValueError(f'Result differs from retained {b.name}')

def run(args):
    owner,evidence,material,ended,capsule,tools,physical,headers=args
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);z=np.load(material)
        cert=json.loads((evidence/'verified-candidate.json').read_text())
        x=z['x']
        if x.dtype!=np.float32 or sha(x)!=cert['input_material_nodes_sha256'] or sha(z['tets'])!=cert['connectivity_sha256'] or sha(z['tags'])!=cert['tags_sha256']:
            raise ValueError('Physical source identities differ')
        with np.load(evidence/'coordinate-delta.npz') as patch:
            y=x.copy();rows=patch['rows'];xor=patch['xor']
            if len(np.unique(rows))!=len(rows) or xor.shape!=(len(rows),3):raise ValueError('Invalid coordinate patch')
            y.view('u4')[rows]^=xor
        if sha(y)!=cert['candidate_material_nodes_sha256']:raise ValueError('Candidate recovery mismatch')
        trial=root/'candidate.npy';np.save(trial,y)
        cpp=Path(__file__).with_name('changed_surface_pairs.cpp')
        exe=root/'check';lib=root/'predicates.dylib'
        flags=['clang++','-std=c++17','-O3','-ffp-contract=off','-I'+str(headers),str(tools/'boundary_pairs.cpp')]
        subprocess.run(flags+[str(cpp),'-o',exe],check=True)
        subprocess.run(flags+['-dynamiclib','-o',lib],check=True)
        fixtures(exe,lib)
        directory=Path(json.loads((ended/'source-binding-parity-036/segments.json').read_text())['myosim_artifact'])
        manifest=directory/'myosim-fullbody-reference.manifest.json'
        new=root/'contact.json'
        compare(owner,manifest,capsule,material,trial,exe,new)
        check_equal(new,evidence/'contact-delta.json')
        paths(owner,manifest,capsule,evidence/'digital-footprints.json',new)
        check_equal(new,evidence/'distal-paths.json')
        allocate=physical/'refit-motion-qualification'
        subprocess.run([sys.executable,tools/'coupled_capacity.py',material,trial,allocate,physical/'close_live_parent_density_compiled',root/'mass.json','--all'],check=True,stdout=subprocess.DEVNULL)
        check_equal(root/'mass.json',evidence/'coupled-mass.json')
        node=root/'coordinates.node';vol=root/'volumes.bin'
        subprocess.run([sys.executable,tools/'export_native_nodes.py',trial,material,node],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([allocate/'read_reference_volumes',node,physical/'constitutive-witness-repaired-tetgen/mesh.ele',vol],check=True)
        if hashlib.sha256(vol.read_bytes()).hexdigest()!=cert['native_volumes_sha256']:raise ValueError('Native volume output differs')
        print('Candidate restored. Differential fixtures, tissue and cortical pairs, terminal paths, typed mass and native volumes reproduced.')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('owner','evidence','material','equilibrium','capsule','registration_tools','physical','headers'):
        parser.add_argument(key,type=Path)
    run(tuple(vars(parser.parse_args()).values()))
