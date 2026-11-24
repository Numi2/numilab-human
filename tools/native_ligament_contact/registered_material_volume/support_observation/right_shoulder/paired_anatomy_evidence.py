"""Compare exact native triangle identities across captures and the independent launcher repeat."""
import json
from pathlib import Path
from collections import Counter
from compile_connected_joint import digest


def read(p):
    return json.loads(p.read_bytes())


def join(folder, repeat, out):
    frames = sorted(folder.glob('step-*.json'),key=lambda f:int(f.stem.split('-')[-1]))
    if len(frames) != 2: raise ValueError('Expected the two retained poses')
    measurements = []
    contacts = []
    selfs = []
    for f in frames:
        d = read(f)
        rep = repeat/'native-traces/accepted-geometry'/(f.stem+'.mrvpack')
        rc = read(rep.with_suffix('.receipt.json'))
        if d['accepted_pack_sha256'] != rc['pack_file_sha256'] or digest(rep) != d['accepted_pack_sha256']:
            raise ValueError('Independent launcher geometry differs')
        pairs = {(r['semantic'],r['stable_id'],r['primitive'],a,b)
                 for r in d['surfaces'] if not r['ocular_source_id'] for a,b in r['triangle_pairs']}
        own = {tuple(sorted(p)) for p in d['self_intersections']['triangle_pairs']}
        if len(pairs) != sum(r['count'] for r in d['surfaces'] if not r['ocular_source_id']) or len(own)!=d['self_intersections']['count']:
            raise ValueError('Repeated witness identity')
        contacts.append(pairs)
        selfs.append(own)
        measurements.append(dict(step=d['accepted_step'], accepted_s=d['accepted_time_s'], pack=rc['pack_file_sha256'],
                                 report=digest(f), self_pairs=len(own), nonocular_pairs=len(pairs), involved_skin_triangles=len({p[3] for p in pairs})))
    keep=contacts[0]&contacts[1];sc=Counter(p[:2] for p in keep)
    result=dict(frames=measurements, persistent_pairs=len(keep), separated_pairs=len(contacts[0]-contacts[1]),
                new_pairs=len(contacts[1]-contacts[0]), persistent_skin_self_pairs=len(selfs[0]&selfs[1]),
                largest_persistent_banks=[dict(semantic=q[0],source=q[1],count=n)for q,n in sc.most_common(10)],
                scope='Exact skin/target face identities at two native captures. Both independently repeated native packs matched byte for byte. Continuous intersection history and solid containment remain unmeasured.')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    import sys
    join(*(Path(s).resolve() for s in sys.argv[1:]))
