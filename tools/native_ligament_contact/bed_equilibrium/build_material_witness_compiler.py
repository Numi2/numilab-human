import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ap = argparse.ArgumentParser()
for name in ('app', 'owner', 'output'):
    ap.add_argument(name, type=Path)
a = ap.parse_args()
a.output.mkdir(exist_ok=True)
src = a.app / 'matter/src/compiler_impl.cpp'
cpp = src.read_text()
loop = '            for (std::size_t sourceTetIndex = 0u; sourceTetIndex < object.tetrahedra.size(); ++sourceTetIndex) {'
start = '                if (referenced && !detail::femReferenceDeterminantInterval(tetrahedron, world.fem.nodes.data())'
end = '                if (framed) {'
last = '            for (std::size_t localNode = 0u; localNode < localMass.size(); ++localNode) {'
assert cpp.count(loop) == cpp.count(start) == cpp.count(last) == 1
cpp = '#include <cstdio>\n' + cpp.replace(loop, '            unsigned materialFailures = 0;\n' + loop)
begin = cpp.index(start)
cpp = cpp[:begin] + '''                if (referenced) {
                    const auto bounds = detail::femReferenceDeterminantInterval(tetrahedron, world.fem.nodes.data());
                    const auto limits = world.materials[elementMaterial].validity;
                    if (!bounds.strictlyAdmitted(limits)) {
                        ++materialFailures;
                        std::fprintf(stderr, "material_cell_witness object=%s cell=%zu material=%u center=%.12g radius=%.12g nominal=%.12g limits=%.12g,%.12g finite=%u bounded=%u nodes=%u,%u,%u,%u\\n",
                            object.name.c_str(), sourceTetIndex, elementMaterial, bounds.center, bounds.radius, double(bounds.nominal),
                            double(limits.x), double(limits.y), unsigned(bounds.finite), unsigned(bounds.noIntermediateOverflow),
                            sourceTet.nodes[0], sourceTet.nodes[1], sourceTet.nodes[2], sourceTet.nodes[3]);
                    }
                }
''' + cpp[cpp.index(end, begin):]
cpp = cpp.replace(last, '''            if (materialFailures) {
                result.diagnostics.push_back({Diagnostic::Severity::error, 0u, 0u,
                    "initial FEM geometry failed executable determinant certification in " + object.name +
                    " (" + std::to_string(materialFailures) + " cells)"});
                return result;
            }
''' + last)
dst = a.output / src.name
dst.write_text(cpp)
build = a.app / 'build-native'
entry = next(e for e in json.loads((build/'compile_commands.json').read_text()) if e['file'] == str(src))
compile = shlex.split(entry['command'])
compile[compile.index('-o')+1] = str(a.output/'compiler.o')
compile[compile.index('-c')+1] = str(dst)
compile[1:1] = ['-I',str(src.parent)]
link = json.loads((a.owner/'identities.json').read_text())['commands'][1]
link[link.index('-o')+1]=str(a.output/'numi-human-native')
link.insert(link.index('lib/libmetalrobo.dylib'),str(a.output/'compiler.o'))
for cmd in (compile,link):
    subprocess.run(cmd,cwd=build,check=True)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(a.output/'identities.json').write_text(json.dumps({'original_compiler_sha256':digest(src),'observed_compiler_sha256':digest(dst),'binary_sha256':digest(a.output/'numi-human-native'),'commands':[compile,link]},indent=2))
