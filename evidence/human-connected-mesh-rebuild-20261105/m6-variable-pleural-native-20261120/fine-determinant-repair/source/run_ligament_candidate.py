from pathlib import Path
prior=Path.cwd().parent/'resting-clearance-and-atrium-010'
source=(prior/'run_settled_ligament_preflight.py').read_text()
source=source.replace("Path(f'settled-ligament-{label}-candidates.json')","(Path.cwd().parent/'resting-clearance-and-atrium-010'/f'settled-ligament-{label}-candidates.json')")
source=source.replace("assert len(rows)==8","shoulder=json.loads(Path('shoulder-region-search.json').read_text())\nassert len(rows)==8 and shoulder['qualified_in_both_poses']\nrow=shoulder['candidate']\nrows[row['name'],row['side']]=row\nassert len(rows)==9")
source=source.replace("Path('prior-aligned-bounded-posture.json')","(here.parent/'resting-clearance-and-atrium-010'/'prior-aligned-bounded-posture.json')")
source=source.replace("timeout=35", "timeout=240")
source=source.replace("paths={str(base/item):str(build/item)","paths={str(base/item):str(build/item)")
source=source.replace("for key in ('NUMI_MATTER_METALLIB'","for path in list(d['pins']):\n    if path != str(executable) and Path(path).name in (app,catalog):\n        del d['pins'][path]\nd['environment']['NUMI_HUMAN_COUPLED_TRACE_CSV']=str(build/'coupled.csv')\nfor key in ('NUMI_MATTER_METALLIB'")
exec(compile(source,"settled_ligament_template", "exec"))
