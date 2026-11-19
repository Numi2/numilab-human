"""Link the native shared owner against the same accepted-step ABI as the app."""
from pathlib import Path
import shlex
import subprocess

from run_support_dependency_preflight import HERE, write, digest

cwd = Path("/Users/numi/Work/skin-fat-native-integration-209/build-native")
isolated = HERE / "shared-fascia-rebase-build" / "lib"
isolated.mkdir(exist_ok=True)
output = isolated / "libmetalrobo.dylib"
runtime = isolated.parent / "runtime.mm.o"
objects = HERE.parent / "resume-fgmres-halfwarp-001" / "objects"
manager = subprocess.check_output(["ninja", "-t", "commands", "lib/libmetalrobo.dylib"],
                                  cwd=cwd, text=True).splitlines()[-1]
link = shlex.split(manager)
assert link[:2] == [":", "&&"] and link[-2:] == ["&&", ":"]
original = "matter/CMakeFiles/numi_matter_runtime_objects.dir/src/runtime.mm.o"
assert link.count(original) == 1 and link.count("lib/libmetalrobo.dylib") == 1
link = [str(output) if arg=="lib/libmetalrobo.dylib"
        else str(runtime) if arg==original
        else str(objects/arg) if arg.endswith(".o")
        else arg for arg in link[2:-2]]
inputs={a:digest(Path(a)) for a in link if a.endswith(".o")}
write(isolated / "link-command.json", {"cwd":str(cwd), "argv":link, "object_sha256":inputs})
subprocess.run(link,cwd=cwd,check=True)
print(digest(output),flush=True)
