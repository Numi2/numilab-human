"""Run the existing accepted-state cell and surface checks using the actual fitted registration."""
import sys,json
from pathlib import Path
from check_full_motion import run
from compile_connected_joint import digest
def qualify(case,volume,fit,stage):
    v=stage/'witness-material/volume';v.mkdir(parents=True,exist_ok=True)
    r=json.loads((fit/'fit.json').read_bytes());cfg=json.loads((case/'launch.json').read_bytes())
    if digest(fit/'current.node')!=r['registered'] or cfg['forearm_registration']!=digest(fit/'fit.json'):raise ValueError('Mismatched physical run')
    for n in ('mesh.node','mesh.ele','material-walls.npz','current.node'):
        f=fit/n if n=='current.node' else volume/n
        if not (v/n).exists():(v/n).symlink_to(f.resolve())
    (v/'result.json').write_text(json.dumps(dict(updated_sites=[dict(vertex=n)for n in r['nodes_0_based']]),indent=2)+'\n')
    return run(case,stage,stage/'motion.json')
if __name__=='__main__':qualify(*(Path(n).resolve()for n in sys.argv[1:]))
