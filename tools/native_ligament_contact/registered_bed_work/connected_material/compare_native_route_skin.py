"""Compare the same body-bound MyoSim landmarks against both registered exteriors."""
import json
from pathlib import Path
import runpy
import sys
import numpy as np
from audit_route_clearance import Surface
from verify_native_extensor_pair import loaded
from run_bed_work import digest


def compare(root,reader):
    d=root/'material-source';registered=json.loads((d/'route-clearance.json').read_text())
    rec,poses,raw=loaded(root/'pose-one',True,False)
    if registered['opening']!=digest(root/'pose-one/terminal.receipt.json'):raise ValueError('Wrong route observation')
    sys.path.insert(0,str(reader.parent/'src'))
    primitives=runpy.run_path(str(reader))['primitives']
    ((_,_,v,f),)=primitives(raw,poses,skin_only=True)
    skin=Surface(v,f);rows={}
    for idx,dense in registered['sites'].items():
        r=skin.query(np.asarray(dense['world_m']))
        rows[idx]=dict(native_side=r['side'],native_distance_m=r['distance_m'],native_axes=r['qualified_axis_count'],dense_side=dense['side'],dense_distance_m=dense['distance_m'])
    bins={}
    for v in rows.values():
        key=v['native_side']+' native / '+v['dense_side']+' dense';bins[key]=bins.get(key,0)+1
    return dict(registered_input=digest(d/'route-clearance.json'),pack=rec['base_pack_file_sha256'],native_open_edges=len(skin.boundary),counts=bins,sites=rows,
                scope='Opening source skin only. Native skin has three measured self-intersection pairs. Parity is interpreted locally, with independently consistent qualified rays; it does not establish global skin validity.')
if __name__=='__main__':
    p=Path(__file__).resolve().parent
    (p/'material-source/native-route-clearance.json').write_text(json.dumps(compare(p,Path(sys.argv[1]).resolve()),indent=2,allow_nan=False)+'\n')
