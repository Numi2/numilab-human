"""Build the isolated trial with one authoritative alpha publisher."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
scratch = Path("/Volumes/NumiHumanTrial")
strict = here / "exact-trial-shader"
out = here / "canonical-glide-alpha-shader"
out.mkdir()
shutil.copytree(strict / "src", out / "src", symlinks=True)
path = out / "src/metal/vascular.metalinc"
original = path.read_text()
start = original.index("kernel void nm_vascular_limit_line_search(")
end = original.index("kernel void nm_vascular_apply_solution(", start)
kernel = original[start:end]
old = """        }
    }
    for (uint object = lane; object < dispatch.objectCount; object += 32u)
        lineSearch[environment * dispatch.objectCount + object].x = alpha;
}
"""
new = """        }
        // Publish the exact same certified FP32 fraction to both consumers.
        // There is one lane owning the scalar and per-object copies.
        const uint objectBase = environment * dispatch.objectCount;
        for (uint object = 0u; object < dispatch.objectCount; ++object)
            lineSearch[objectBase + object].x = alpha;
    }
}
"""
assert kernel.count(old) == 1
updated = original[:start] + kernel.replace(old, new) + original[end:]
path.unlink()
path.write_text(updated)
(here / "canonical-glide-alpha.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    "a/matter/src/metal/vascular.metalinc", "b/matter/src/metal/vascular.metalinc")))
commands = json.loads((strict / "build-commands.json").read_text())
commands[0] = [arg.replace(str(strict / "src"), str(out / "src")) for arg in commands[0]]
intermediate = scratch / "canonical-glide-alpha.air"
library = scratch / "canonical-glide-alpha.metallib"
commands[0][-1] = str(intermediate)
commands[1][-3], commands[1][-1] = str(intermediate), str(library)
(out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for command in commands:
    subprocess.run(command, check=True)
shutil.copyfile(library, out / "NumiMatter.metallib")
intermediate.unlink()
library.unlink()
