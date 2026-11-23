"""Submit the refined connected shoulders, current material labels and actual donor masses to the production native material importer."""
from pathlib import Path
import sys,json,os
import numpy as np
from build_carpal_cavity import boundary
from verify_carpal_mesh import classify
from partition_carpal_material import solid
from compile_connected_joint import digest
from import_wrist_cavity import run as import_domain


def run(parent,domain,geom,flow,alloc,mass,work):
    g=json.loads(geom.read_text());m=json.loads(mass.read_text());z=np.load(domain)
    if digest(domain)!=m['domain'] or digest(domain)!=g['mesh'] or any(v['low'] or any(w!=0 for w in v['overlap_mm3'].values()) for v in g['layers'].values()):
        raise ValueError('Missing all-cortex and cell-quality admission')
    built=json.loads((parent/'result.json').read_text())
    if built['exit'] or any(digest(parent/k)!=sha for k,sha in built['output_hashes'].items()):raise ValueError('Unverified parent material')
    work.mkdir();mesh=work/'source';mesh.mkdir();maps=work/'receiver';maps.mkdir()
    for k in built['output_hashes']:
        (mesh/k).hardlink_to(parent/k)
    (mesh/'receipt.json').hardlink_to(parent/'result.json')
    (mesh/'native-import-build.json').write_text(json.dumps(dict(binary=built['binary'],launch=built['command'])))
    x=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4')
    labels,outer=classify(z['x'].astype(float),z['t'],z['facets'],z['markers'])
    if not np.array_equal(labels,z['role']):raise ValueError('Lost a material interface')
    def keys(pts,faces):return {tuple(sorted(pts[row],key=lambda s:s.tobytes())[i].tobytes() for i in range(3)) for row in faces}
    expected=keys(x,z['fence']);actual=keys(z['x'],outer);missing=len(expected-actual)
    if missing:raise ValueError(f'{missing} missing native joining faces')
    origin=z['x'].astype(float).mean(0);sol=[]
    for mat in (15,16):
        faces=boundary(z['t'][labels==mat]);idx,ix=np.unique(faces,return_inverse=True)
        sol.append(solid((z['x'][idx].astype(float)-origin)*1000,ix.reshape(-1,3)))
    common=sol[0]^sol[1]
    if common.status()!=sol[0].status() or abs(common.volume())>1.e-8:raise ValueError(f'Skin/fat solid overlap {common.volume()} mm3')
    native_cells=len(np.fromfile(mesh/'cell-mass.f64',dtype='<f8'))
    report={};mat={}
    for tag,role in [('SKIN',15),('FAT',16)]:
        if not m[tag]['passed_mass'] or m[tag]['invalid_cells'] or not json.loads((alloc/(tag+'-transport.json')).read_text())['final']['passed']:raise ValueError('Unqualified receiver')
        a=np.load(alloc/(tag+'-transport.npz'));rec=np.r_[a['external'],native_cells+a['child_ids']]
        val=np.r_[a['external_mass_kg'],a['child_mass_kg']]
        if len(np.unique(rec))!=len(rec) or not np.all(val>0):raise ValueError('Nonpositive or duplicate receiver')
        np.savez_compressed(maps/(tag+'-mapping.npz'),recipient=rec,receiver_mass_kg=val)
        report[tag]=dict(passed=True,transport=digest(alloc/(tag+'-transport.npz')),full_mass=digest(mass))
        mat[role]=dict(invalid=g['layers'][str(role)]['low'],cortex_intersection_mm3=sum(g['layers'][str(role)]['overlap_mm3'].values()))
    (maps/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    proof=work/'geometry.json';proof.write_text(json.dumps(dict(candidate_sha256=digest(domain),missing_joining_faces=missing,joining_faces=len(expected),skin_fat_intersection_mm3=common.volume(),materials=mat),indent=2)+'\n')
    import_domain(mesh,domain,flow/'ancestry',maps,proof,work/'native')
    native=json.loads((work/'native/native-import.json').read_text())
    if not native['native_imported'] or any(native[k] for k in ('inverted','below_shape','below_volume','nonfinite','density_outliers','zero_mass_nodes')):raise ValueError(f'Native rejection {native}')
    if any(abs(native[k]-m[tag]['source_kg'])>1e-11 for k,tag in [('skin_kg','SKIN'),('fat_kg','FAT')]):raise ValueError('Native mass differs')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
