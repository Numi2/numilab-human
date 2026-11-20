"""Regenerate the existing coarse skin source registration from exact native captures."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_literal(path, key):
    match = re.search(r'\b'+key+r'\s*=\s*\{(.*?)\};', path.read_text(), re.S)
    if not match:
        raise ValueError(f'missing {key}: {path}')
    value = re.sub(r'//[^\n]*', '', match[1]).strip('{} \n')
    value = value.replace('std::numeric_limits<std::uint32_t>::max()', '-1')
    value = value.replace('kWholeBodySkinTransferUnassigned', '-1')
    return np.fromstring(re.sub(r'std::array<double,3>|Face|[{}fu]', '', value).replace(',', ' '), sep=' ')


def oriented(face):
    a,b,c = map(int, face)
    return min((a,b,c), (b,c,a), (c,a,b))


def load_capture(path, asset):
    rows = np.loadtxt(path, delimiter=',', skiprows=1)
    if rows.shape != (asset['nv'],20) or not np.isfinite(rows).all():
        raise ValueError(f'incomplete or nonfinite native observation: {path}')
    if not np.array_equal(rows[:,0], np.arange(asset['nv'])):
        raise ValueError(f'mismatched vertex IDs: {path}')
    body_ids = asset['binds']['body']
    if not np.array_equal(rows[:,1], body_ids[np.argmax(asset['weights'], axis=1)]):
        raise ValueError(f'nonmatching native owner IDs: {path}')
    return rows


def build_rebind(old, new, before, after, source_vertices, faces, beta):
    size = len(source_vertices)
    new_faces = np.full(size, -1, dtype=np.int64)
    new_beta = np.zeros((size,3))
    old_source_points = np.empty((size,3))
    new_source_points = np.empty((size,3))
    new_normals = np.empty((size,3))
    new_weights = np.empty((size,new['nb']))
    new_by_key = {}
    for i,face in enumerate(new['inds']):
        key = oriented(face)
        new_by_key.setdefault(key, []).append(i)

    for row, source in enumerate(source_vertices):
        if source >= 0:
            if faces[row] >= 0 or source >= new['nv']:
                raise ValueError(f'invalid direct identity: {row}')
            old_source_points[row] = before[source,2:5]
            new_source_points[row] = after[source,2:5]
            new_normals[row] = after[source,5:8]
            new_weights[row] = new['weights'][source]
            continue

        face = faces[row]
        weights = beta[row]
        if (not 0 <= face < len(old['inds']) or np.min(weights) < -1e-8 or
            abs(weights.sum()-1) > 1e-8):
            raise ValueError(f'invalid coarse source projection: {row}')
        previous = old['inds'][face]
        matches = new_by_key.get(oriented(previous), [])
        if len(matches) != 1:
            raise ValueError(f'coarse row {row} source face {face} needs explicit local ancestry')
        index = matches[0]
        current = new['inds'][index]
        reordered = np.array([weights[np.flatnonzero(previous == corner)[0]] for corner in current])
        new_faces[row] = index
        new_beta[row] = reordered
        old_source_points[row] = weights @ before[previous,2:5]
        new_source_points[row] = reordered @ after[current,2:5]
        new_normals[row] = reordered @ after[current,5:8]
        new_weights[row] = reordered @ new['weights'][current].astype(float)

    assert np.max(np.abs(new_weights.sum(axis=1)-1)) < 5e-7
    lengths = np.linalg.norm(new_normals, axis=1)
    assert np.min(lengths) > 1e-8
    new_normals /= lengths[:,None]
    return new_faces, new_beta, old_source_points, new_source_points, new_normals, new_weights


def signed_prisms(outer, inner, source_triangles):
    orientation = np.ones(len(source_triangles))
    for i,j in ((0,1),(0,2),(1,2)):
        orientation *= np.where(source_triangles[:,i] > source_triangles[:,j], -1., 1.)
    a,b,c = np.sort(source_triangles, axis=1).T

    def det6(x,y,z,w):
        return np.einsum('ij,ij->i', y-x, np.cross(z-x,w-x))

    # Match the native sorted-row prism split and source winding exactly.
    return np.stack((det6(outer[a],outer[b],outer[c],inner[c]),
                     -det6(outer[a],outer[b],inner[b],inner[c]),
                      det6(outer[a],inner[a],inner[b],inner[c])), axis=1) * (orientation[:,None]/6.)


def summarize_volumes(outer, inner, faces, depth):
    actual = signed_prisms(outer.astype(np.float32).astype(float),
                           inner.astype(np.float32).astype(float), faces)
    tolerance = np.maximum(1e-8, depth[faces].max(axis=1))**3 * 1e-12
    invalid = np.argwhere(actual <= tolerance[:,None])
    return {'cells':int(actual.size), 'invalid_cells':len(invalid), 'witnesses':invalid.tolist(),
            'minimum_oriented_volume_m3':float(actual.min()),
            'signed_volume_m3':float(actual.sum())}, actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner',type=Path,required=True)
    parser.add_argument('--identity',type=Path,required=True)
    parser.add_argument('--frozen-capture',type=Path,required=True)
    parser.add_argument('--current-capture',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.owner/'tools'))
    from rebind_human_skin_reference import decode

    identities = json.loads(args.identity.read_text())
    old_path, new_path = [Path(identities[k]['path']) for k in ('frozen_source','current')]
    pins = {str(p):sha256(p) for p in (old_path,new_path,args.frozen_capture,args.current_capture)}
    assert pins[str(old_path)] == identities['frozen_source']['sha256']
    assert pins[str(new_path)] == identities['current']['sha256']
    old,new = [decode(p.read_bytes()) for p in (old_path,new_path)]
    assert old['nv']==new['nv'] and old['fingerprint']==new['fingerprint']
    assert old['source_sha256']==new['source_sha256'] and np.array_equal(old['binds'],new['binds'])

    before,after = load_capture(args.frozen_capture,old),load_capture(args.current_capture,new)
    unchanged = np.all(old['verts'][:,:3]==new['verts'][:,:3],axis=1) & np.all(old['weights']==new['weights'],axis=1)
    matched_error = float(np.max(np.linalg.norm(before[unchanged,2:5]-after[unchanged,2:5],axis=1)))
    if matched_error > 2e-6 or not np.array_equal(before[unchanged,11:],after[unchanged,11:]):
        raise ValueError(f'native captures use different skeletal frames: {matched_error:.9g} m')

    reduction = args.owner/'apps/NumiHumanWholeBodySkinReduction.hpp'
    ancestry = args.owner/'apps/NumiHumanWholeBodySkinTransferAncestry.hpp'
    candidate = args.owner/'apps/NumiHumanFinal128CandidateRegistration.hpp'
    for path in (reduction,ancestry,candidate):
        pins[str(path)]=sha256(path)
        assert pins[str(old_path)] in path.read_text()
    src = read_literal(reduction,'kWholeBodySkinSourceVertices').astype(np.int64)
    rows= len(src)
    face = read_literal(ancestry,'kWholeBodySkinSourceEmbeddingFaces').astype(np.int64)
    beta = read_literal(ancestry,'kWholeBodySkinSourceEmbeddingBarycentric').reshape(rows,3)
    residual = read_literal(ancestry,'kWholeBodySkinSourceEmbeddingResidualWorldM').reshape(rows,3)
    coarse_triangles = read_literal(reduction,'kWholeBodySkinFaces').astype(np.int64).reshape(-1,3)
    reference = read_literal(candidate,'kWholeBodySkinCandidateReferenceWorldPoints').reshape(rows,3)
    new_face, new_beta, old_points, new_points, normals, weights = build_rebind(
        old,new,before,after,src,face,beta)
    direct=src>=0
    rest_points = old_points[direct]
    ref_mean = reference[direct].mean(axis=0)
    pose_mean = rest_points.mean(axis=0)
    u,_,v = np.linalg.svd((rest_points-pose_mean).T @ (reference[direct]-ref_mean))
    rotation = u @ v
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError('source registration would reflect anatomical laterality')
    placement_error = float(np.max(np.linalg.norm((old_points-pose_mean)@rotation+residual+ref_mean-reference,axis=1)))
    if placement_error > 2e-6:
        raise ValueError(f'frozen reference ancestry cannot be reproduced: {placement_error:.9g} m')
    delta=(new_points-old_points)@rotation
    proposed_reference = reference + delta
    dominant = new['binds']['body'][weights.argmax(axis=1)]
    old_dominant = read_literal(candidate,'kWholeBodySkinCandidateDonorBodies').astype(np.int64)
    donor_frame=np.empty((rows,9))
    donor_local=np.empty((rows,3))
    for body in np.unique(dominant):
        source_donors=after[after[:,1]==body]
        if not len(source_donors):
            raise ValueError(f'no observed frame for {body}')
        frame=source_donors[0,11:].reshape(3,3)
        if not np.all(source_donors[:,11:]==frame.ravel()):
            raise ValueError(f'native frame is not constant for {body}')
        donor_origin=np.median(source_donors[:,2:5]-np.linalg.solve(frame,source_donors[:,8:11].T).T,axis=0)
        use=dominant==body
        donor_frame[use]=(frame @ rotation).ravel()
        in_pose=(proposed_reference[use]-ref_mean)@rotation.T+pose_mean
        donor_local[use]=(in_pose-donor_origin) @ frame.T
    normals=normals@rotation

    correction = read_literal(reduction,'kWholeBodySkinMechanicalWorldDeltas').reshape(rows,3)
    inner = read_literal(reduction,'kWholeBodySkinInnerWorld').reshape(rows,3)
    deep = read_literal(reduction,'kWholeBodyFatDeepWorld').reshape(rows,3)
    depth = np.stack((read_literal(reduction,'kWholeBodySkinThicknessM'),
                      read_literal(reduction,'kWholeBodyFatThicknessM')))
    # Exact source displacement transfer by material column. Keep both depth
    # vectors unchanged. Positivity is a gate, not an enforced sign flip.
    previous_layers=(reference+correction,inner,deep)
    proposed_layers=(proposed_reference+correction,inner+delta,deep+delta)
    metrics={}
    for i,(name,density) in enumerate((('skin',1000.),('fat',900.))):
        original, old_volume = summarize_volumes(*previous_layers[i:i+2],coarse_triangles,depth[i])
        trial, volume = summarize_volumes(*proposed_layers[i:i+2],coarse_triangles,depth[i])
        trial['density_kg_m3']=density
        trial['geometry_mass_kg_if_valid']=float(density*volume.sum()) if trial['invalid_cells']==0 else None
        trial['new_negative_or_degenerate_cells']=int(np.count_nonzero((volume <= 0)&(old_volume > 0)))
        metrics[name]={'reference':original,'proposal':trial}

    args.out.mkdir(exist_ok=False)
    np.savez_compressed(args.out/'registration.npz',source_vertex=src,source_face=new_face,
                        barycentric=new_beta,projected_residual_world_m=residual,donor_body=dominant,
                        world_m=proposed_reference,normal_world=normals,donor_local_m=donor_local,
                        world_to_donor=donor_frame,full86_weights=weights)
    report = {'inputs_sha256':pins,'registration_sha256':sha256(args.out/'registration.npz'),
              'rows':rows,'direct_rows':int(np.count_nonzero(src>=0)),
              'projected_rows':int(np.count_nonzero(src<0)),
              'remapped_projected_faces':int(np.count_nonzero(face!=new_face)),
              'owner_changes':np.flatnonzero(dominant!=old_dominant).tolist(),
              'native_pose_max_error_m':matched_error,'frozen_max_registration_error_m':placement_error,
              'mechanical_source_rows_changed_over_1um':int(np.count_nonzero(np.linalg.norm(delta,axis=1)>1e-6)),
              'maximum_mechanical_source_delta_m':float(np.linalg.norm(delta,axis=1).max()),
              'paired_prism_diagnostic':metrics,'runtime_admitted':False}
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','paired_prism_diagnostic')}))
    for layer, values in metrics.items():
        print(layer,{stage:{k:v for k,v in data.items() if k!='witnesses'} for stage,data in values.items()})


if __name__=='__main__':
    main()
