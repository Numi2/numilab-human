"""Locate bilateral native force sites relative to their own rigid cortical surfaces."""
import argparse
import ctypes as C
import hashlib
import subprocess
import json
from pathlib import Path
import runpy
import struct
import sys
import numpy as np
from material_context import sha
from measure_forearm_enclosure import Enclosure
from verify_native_extensor_pair import loaded


def measure(run, routes, reader, library, out, compact=False):
    receipt,pose,raw=loaded(run,False,not compact)
    data=json.loads(routes.read_text())
    if receipt['pack_file_sha256']!=data['pack']:raise ValueError('Unmatched force routes')
    sys.path.insert(0,str(reader.parent/'src'));audit=runpy.run_path(str(reader))
    h=audit['clearance']._PACK_HEADER;d=audit['clearance']._PACK_DIRECTORY
    sections={r[0]:r for r in (d.unpack_from(raw,h.size+i*d.size) for i in range(h.unpack_from(raw)[2]))}
    prim,inst=sections[4],sections[5]
    if compact:
        source = subprocess.check_output(['zstd', '-qdc', run / 'initial.mrvpack.zst'])
    else:
        source = (run / 'native-traces/resting-human.mrvpack').read_bytes()
    if hashlib.sha256(source).hexdigest() != receipt['base_pack_file_sha256']:
        raise ValueError('Unbound rigid source')
    base_sections={r[0]:r for r in (d.unpack_from(source,h.size+i*d.size) for i in range(h.unpack_from(source)[2]))}
    bp,bi=base_sections[4],base_sections[5]
    if bp[4]!=prim[4]:raise ValueError('Primitive table changed size')
    rows=[dict(s,member=r['name']) for r in data['banks'] for s in r['sites']]
    by={}
    for i,s in enumerate(rows):by.setdefault(s['body'],[]).append(i)
    (_,_,x,f),=audit['primitives'](raw,pose,skin_only=True)
    skin=Enclosure(library,x,f)
    closest={}
    try:
        for (semantic,stable),index,x,f in audit['primitives'](raw,pose):
            if semantic!=51004:continue
            record=struct.unpack_from('<8I',raw,prim[2]+64*index)
            transform=struct.unpack_from('<8f12I',raw,inst[2]+80*record[3])
            body=transform[9]
            base_record=struct.unpack_from('<8I',source,bp[2]+64*index)
            bone=struct.unpack_from('<8f12I',source,bi[2]+80*base_record[3])
            if tuple(base_record[:2])!=tuple(record[:2]) or base_record[4]!=semantic or base_record[7]!=stable or bone[9]!=body or bone[10]!=3 or transform[10] not in (0,3):
                raise ValueError('Baked cortical instance lost its registered owner')
            if body not in by:continue
            surface=Enclosure(library,x,f)
            try:
                points=np.ascontiguousarray([rows[i]['executed_world_m'] for i in by[body]],dtype='f8')
                labels,distances=surface.query(points)
                ptr=np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS')
                ip=np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
                query=surface.lib.cortex_query;query.argtypes=[C.c_void_p,ptr,C.c_int,ptr,ip];query.restype=None
                results=np.empty((len(points),7),'f8');ids=np.empty(len(points),'i4')
                query(surface.handle,points,len(points),results,ids)
                if not np.allclose(np.abs(results[:,0]),distances,rtol=1e-10,atol=1e-12):raise ValueError('Nearest cortical surface mismatch')
                position=results[:,4:7];foot_label,foot_dist=skin.query(position)
                for idx,i in enumerate(by[body]):
                    mm=distances[idx]*1000
                    if i not in closest or mm<closest[i]['cortical_distance_mm']:
                        closest[i]=dict(cortical_stable_id=stable,primitive=index,cortical_distance_mm=float(mm),
                            site_cortical_enclosure=int(labels[idx]),nearest_cortex_skin_enclosure=int(foot_label[idx]),
                            nearest_cortex_skin_depth_mm=float(1000*foot_dist[idx]),nearest_world_m=position[idx].tolist())
            finally:surface.close()
    finally:skin.close()
    if set(closest)!=set(range(len(rows))):raise ValueError('Missing declared rigid cortex for a force site')
    for i,row in enumerate(rows):row.update(closest[i])
    out.write_text(json.dumps(dict(pack=data['pack'],myosim=data['myosim'],tendon=data['tendon'],source_routes=sha(routes),
        scope='Distance of each named force site to the nearest surface owned by its declared rigid body. The nearest point is a diagnostic landmark.',sites=rows),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'routes', 'reader', 'library', 'out'):
        parser.add_argument(name, type=lambda v: Path(v).resolve())
    parser.add_argument('--compact', action='store_true', help='read the supported-motion replay')
    measure(**vars(parser.parse_args()))
