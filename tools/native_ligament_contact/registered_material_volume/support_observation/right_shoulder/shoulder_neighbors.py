"""Measure a candidate against all native neighboring muscle, bone and passive organ solids."""
import json,mmap,runpy
from pathlib import Path
import sys
import numpy as np
from prepare_material_interface import solid
from compile_connected_joint import digest


def inspect(case,registration,candidate,reader,out):
    z=np.load(registration);u=np.load(candidate)['displacement_m'];x=z['accepted'];origin=x.mean(0)
    f=z['boundary'];b={k:solid((pos-origin)*1000,f,k)[0] for k,pos in [('before',x),('after',x+u)]}
    lo=np.minimum(x.min(0),(x+u).min(0));hi=np.maximum(x.max(0),(x+u).max(0))
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'));r=json.loads(rc.read_bytes())
    pack=Path(r['accepted_pack_path']);assert digest(pack)==r['pack_file_sha256']
    poses={a['body_index']:a for a in r['accepted_body_poses']};api=runpy.run_path(str(reader));rows=[]
    with pack.open('rb')as s,mmap.mmap(s.fileno(),0,access=mmap.ACCESS_READ)as buf:
        for (kind,stable),_,v,t in api['primitives'](buf,poses):
            if (kind,stable)==(51005,83) or kind==51007:continue
            if np.any(v.max(0)<lo) or np.any(v.min(0)>hi):continue
            m,_,meta=solid((v-origin)*1000,t,'neighbor')
            if str(m.status())!='Error.NoError' or m.is_empty() or meta['mesh']['nonmanifold_or_open']:
                rows.append(dict(kind=int(kind),bank=int(stable),indeterminate=str(m.status()),topology=meta['mesh']));continue
            row=dict(kind=int(kind),bank=int(stable))
            for k,a in b.items():row[k]=float(abs((a^m).volume()))
            rows.append(row)
    out.write_text(json.dumps(dict(registration=digest(registration),candidate=digest(candidate),capture=digest(rc),units='mm3',
            scanned_classes=sorted(api['BANK_CLASSES']),excluded_outer_skin='Separate material enclosure check required',surrounding=rows),indent=2)+'\n')
    print([(r['kind'],r['bank'],round(r['before'],2),round(r['after'],2))for r in rows if 'before'in r and max(r['before'],r['after'])>.001])
if __name__=='__main__':inspect(*(Path(s).resolve()for s in sys.argv[1:]))
