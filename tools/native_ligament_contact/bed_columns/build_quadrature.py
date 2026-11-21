from pathlib import Path
import hashlib, json, shutil, subprocess, sys

baseline, app, dest = map(Path, sys.argv[1:4])
here = Path(__file__).resolve().parent
dest.mkdir(exist_ok=True)
for f in (*baseline.glob("*.hpp"), *baseline.glob("*.mm")):
    shutil.copyfile(f, dest / f.name)
header = dest / "NumiHumanRestingVisual.hpp"
s = header.read_text()
def edit(old, new):
    global s
    if s.count(old) != 1:
        raise ValueError(f"Expected a unique source site: {old}")
    s = s.replace(old, new)
edit('#pragma once', '#pragma once\n#include "NumiHumanBedColumnQuadrature.hpp"')
edit('std::vector<double> skinThicknessByRow', 'std::vector<NumiHumanBedColumnQuadrature> materialColumns(regions.size());\n            std::vector<double> skinThicknessByRow')
edit('skinThicknessByRow[row] += tributaryAreas[v]', 'materialColumns[row].add(tributaryAreas[v], kWholeBodySkinThicknessM[nearest],\n                                             kWholeBodyFatThicknessM[nearest]);\n                    skinThicknessByRow[row] += tributaryAreas[v]')
edit('within_region_nearest_then_projected_area_average','within_region_nearest_parallel_material_columns')
edit('constexpr double skinModulus = 200000.0;', 'constexpr double skinModulus = NumiHumanBedColumnQuadrature::skinModulusPa;')
edit('constexpr double fatModulus = 55000.0;', 'constexpr double fatModulus = NumiHumanBedColumnQuadrature::fatModulusPa;')
edit('NumiHumanLocalBedSection{float(skinThickness/(skinModulus * areas[r])),\n                        float(fatThickness/(fatModulus * areas[r])), retardation,',
     '''NumiHumanLocalBedSection{float(useRegionalThickness ? materialColumns[r].skinComplianceMPerN()
                                                                    : skinThickness / (skinModulus * areas[r])),
                        float(useRegionalThickness ? materialColumns[r].fatComplianceMPerN()
                                                   : fatThickness / (fatModulus * areas[r])), retardation,''')
header.write_text(s)
shutil.copyfile(here / 'NumiHumanBedColumnQuadrature.hpp', dest / 'NumiHumanBedColumnQuadrature.hpp')
(dest / 'metalrobo').mkdir(exist_ok=True)
for metal in (baseline / 'metalrobo').glob('*.metallib'):
    target = dest / 'metalrobo' / metal.name
    if not target.exists():
        shutil.copy2(metal, target)
    if metal.read_bytes() != target.read_bytes():
        raise ValueError('Mismatched executable-relative runtime shader')
cmds = json.loads((baseline / 'build-commands.json').read_text())
for cmd in cmds:
    for i, arg in enumerate(cmd):
        if Path(arg).parent == baseline:
            cmd[i] = str(dest / Path(arg).name)
    subprocess.run(cmd, cwd=app / "build-native", check=True)
(dest / 'build-commands.json').write_text(json.dumps(cmds, indent=2) + '\n')
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
(dest / "built.json").write_text(json.dumps({str(p):digest(p) for p in [baseline / header.name, header, dest / 'NumiHumanBedColumnQuadrature.hpp', dest / 'numi-human-native']},indent=2)+'\n')
