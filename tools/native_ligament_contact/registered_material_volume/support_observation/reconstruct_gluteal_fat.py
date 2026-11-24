"""Construct the deep adipose volume over the loaded gluteal surface basis functions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from bind_gluteal_receivers import digest, require


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def signed_area(polygon):
    a = polygon[1:-1] - polygon[0]
    b = polygon[2:] - polygon[0]
    return 0.5 * cross2(a, b).sum()


def halfplane(poly, distances):
    values = []
    for i, point in enumerate(poly):
        a, b = distances[i-1], distances[i]
        if a*b < 0:
            values.append(poly[i-1] + (a/(a-b)) * (point-poly[i-1]))
        if b >= 0:
            values.append(point)
    return np.asarray(values).reshape(-1,2)


def side(poly, hull, index, orientation):
    start = hull[index]
    return orientation * cross2(hull[(index+1) % len(hull)]-start, poly-start)


def has_area(poly):
    return len(poly)>=3 and abs(signed_area(poly))>1e-16


def clip(poly, hull):
    orientation = np.sign(signed_area(hull))
    for i in range(len(hull)):
        if not has_area(poly):
            return np.empty((0,2))
        poly = halfplane(poly, side(poly,hull,i,orientation))
    return poly


def subtract(poly, cover):
    """Convex parts of a convex polygon outside a convex occluder."""
    overlap=clip(poly, cover)
    if not has_area(overlap):
        return [poly]
    result=[]
    orientation=np.sign(signed_area(cover))
    for i in range(len(cover)):
        if not has_area(poly):
            break
        d=side(poly,cover,i,orientation)
        piece=halfplane(poly,-d)
        if has_area(piece):
            result.append(piece)
        poly=halfplane(poly,d)
    return result


def visible_parts(poly, base, top, shields):
    pieces=[poly]
    for shield in shields:
        cut=clip(poly,shield[:,:2])
        if not has_area(cut):
            continue
        heights=lambda f, xy: bary(f,xy) @ f[:,2]
        cut=halfplane(cut, heights(top,cut)-heights(shield,cut))
        if not has_area(cut):
            continue
        cut=halfplane(cut, heights(shield,cut)-heights(base,cut))
        if has_area(cut):
            pieces=[p for patch in pieces for p in subtract(patch,cut)]
        if not pieces:
            break
    return pieces


def bary(face, points):
    uv = np.linalg.solve((face[1:, :2] - face[:1, :2]).T, (points - face[0, :2]).T).T
    return np.column_stack((1 - uv.sum(axis=1), uv))


def box_candidates(faces, lo, hi):
    return np.flatnonzero(np.all(faces.max(1)[:, :2] >= lo, axis=1) &
                          np.all(faces.min(1)[:, :2] <= hi, axis=1))


def glutes(assets, capture, receivers, frame, origin):
    manifest = json.loads((assets/'manifest.json').read_bytes())
    require(digest(assets/'manifest.json') == receivers['atlas_sha256'], 'Unknown gluteal atlas')
    def table(name, kind):
        f = assets / (name + '.bin')
        require(digest(f) == manifest['files'][f.name], 'Changed atlas table ' + name)
        return np.fromfile(f, kind).reshape(-1, 3, 4)
    data = table('constitutive-boundary-reference', '<f4').astype(float)
    local = data[:, 0, :3] + data[:, 1, :3]
    pelvis = next(r for r in capture['accepted_body_poses'] if r['body_index'] == 128)
    world = Rotation.from_quat(pelvis['quaternion_xyzw']).apply(local) + pelvis['position_m']
    faces = table('closed-material-triangles', '<u4')[:, 0, :3]
    xyz = ((world-origin) @ frame)[faces]
    owner = np.empty(len(faces), int)
    for spec, span in zip(manifest['contact_flexibility'], manifest['closed_reverse']['sides']):
        owner[span['first_face']:span['first_face']+span['faces']] = spec['muscle']
    return xyz, owner


def propose(prior, volume, assets, out):
    cfg = json.loads((prior/'coupled/launch.json').read_bytes())
    pose = prior/'coupled/native-traces/accepted-geometry/step-1.receipt.json'
    captures = json.loads(pose.read_bytes())
    sites = json.loads((prior/'registered-receivers.json').read_bytes())
    rays = json.loads((prior/'current-rays.json').read_bytes())
    physical = prior/'coupled/physical-accepted.node'
    require(digest(physical) == rays['origin_frame'] and digest(pose) == sites['pose'] == rays['pose'],
            'Unmatched supported state')
    x = np.loadtxt(physical, skiprows=1, usecols=(1,2,3))
    walls = np.load(volume/'material-walls.npz')
    n = np.asarray(rays['bed_normal'])
    e = np.cross([0,1,0], n)
    e /= np.linalg.norm(e)
    frame = np.column_stack((e, np.cross(n,e), n))
    nodes = {r['skin_node']:r['muscle'] for r in sites['receivers']}
    origin = x[list(nodes)].mean(axis=0)
    xyz = (x-origin) @ frame
    lower = xyz[walls['D']]
    upper, muscle = glutes(assets, captures, sites, frame, origin)
    selected = np.flatnonzero(np.isin(walls['O'], list(nodes)).any(1))
    lower_ids = box_candidates(lower, xyz[walls['O'][selected]].min((0,1))[:2], xyz[walls['O'][selected]].max((0,1))[:2])
    upper_ids = box_candidates(upper, xyz[walls['O'][selected]].min((0,1))[:2], xyz[walls['O'][selected]].max((0,1))[:2])
    # Only entering muscle surfaces face the supporting bed.
    upper_ids = upper_ids[cross2(upper[upper_ids,1,:2]-upper[upper_ids,0,:2],
                                upper[upper_ids,2,:2]-upper[upper_ids,0,:2]) < 0]
    # Registration may contain additional intersecting surfaces. Count all projected
    # coverings and refuse a patch that does not have exactly one physical interval.
    columns = []
    coverage = []
    for outer in selected:
        tri = xyz[walls['O'][outer]]
        requested = {nodes[int(i)] for i in walls['O'][outer] if int(i) in nodes}
        require(len(requested) == 1, 'Ambiguous bilateral footprint')
        sid = requested.pop()
        foot = tri[:, :2]
        ref_area = abs(signed_area(foot))
        total_area = 0.
        li = lower_ids[box_candidates(lower[lower_ids], foot.min(0), foot.max(0))]
        ui = upper_ids[box_candidates(upper[upper_ids], foot.min(0), foot.max(0))]
        ui = ui[muscle[ui] == sid]
        for lid in li:
            base = lower[lid]
            p = clip(foot, base[:, :2])
            if len(p) < 3 or abs(signed_area(p)) < 1e-16:
                continue
            for uid in ui:
                top = upper[uid]
                poly = clip(p, top[:, :2])
                if len(poly) < 3 or abs(signed_area(poly)) < 1e-16:
                    continue
                b0, b1 = bary(base, poly), bary(top, poly)
                z0, z1 = b0 @ base[:,2], b1 @ top[:,2]
                skin = bary(tri, poly) @ tri[:,2]
                if np.max(z1-z0) <= 0 or np.max(z0-skin) <= 0:
                    continue
                require(np.min(z1-z0)>0 and np.min(z0-skin)>0, 'Surface ordering changes within footprint')
                candidates=ui[ui!=uid]
                if len(candidates):
                    lo,hi=poly.min(0),poly.max(0)
                    candidates=candidates[box_candidates(upper[candidates],lo,hi)]
                    candidates=candidates[upper[candidates,:,2].min(1)<z1.max()]
                for part in visible_parts(poly,base,top,upper[candidates]):
                    b0,b1=bary(base,part),bary(top,part)
                    z0,z1=b0 @ base[:,2],b1 @ top[:,2]
                    for i in range(1,len(part)-1):
                        ix=[0,i,i+1]
                        area=abs(signed_area(part[ix]))
                        if area<=1e-16:
                            continue
                        total_area+=area
                        columns.append((int(outer),sid,int(lid),int(uid),area,part[ix],z0[ix],z1[ix],b0[ix],b1[ix]))
        coverage.append((int(outer),float(ref_area),float(total_area)))
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'registered-columns.npz', outer=[c[0] for c in columns], muscle=[c[1] for c in columns],
        deep_fat_face=[c[2] for c in columns], muscle_face=[c[3] for c in columns], projected_area=[c[4] for c in columns],
        plane_xy=[c[5] for c in columns], fat_z=[c[6] for c in columns], muscle_z=[c[7] for c in columns],
        fat_shape=[c[8] for c in columns], muscle_shape=[c[9] for c in columns], frame=frame, origin=origin)
    areas=np.array(coverage)
    summary={'physical_nodes':digest(physical),'wall_topology':digest(volume/'material-walls.npz'),
             'native_pose':digest(pose),'registered_gluteal_atlas':sites['atlas_sha256'],
             'registered_bed':digest(Path(cfg['argv'][cfg['argv'].index('--support-contact-payload')+1])),
             'columns':len(columns),'outer_basis_faces':len(selected),'coverage':coverage,
             'projected_reference_area_m2':float(areas[:,1].sum()),'projected_covered_area_m2':float(areas[:,2].sum()),
             'maximum_relative_area_error':float(max(abs(areas[:,2]/areas[:,1]-1))),
             'volume_m3':sum(c[4]*np.mean(c[7]-c[6]) for c in columns),
             'source':'Inferred deep gluteal adipose between registered surfaces. The source330 tissue volume ends at the deep-fat face.',
             'columns_sha256':digest(out/'registered-columns.npz')}
    (out/'registration.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k in ('columns','outer_basis_faces','maximum_relative_area_error','volume_m3')})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('prior','volume','assets','out'):
        parser.add_argument(key,type=Path)
    args=parser.parse_args()
    propose(**vars(args))
