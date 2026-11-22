import json
from pathlib import Path
import subprocess
import sys
from material_context import sha
from patch_native import metal, host


def build(base, library, body, templates, out):
    out.mkdir(exist_ok=True)
    dst = out / "src"
    dst.mkdir(exist_ok=True)
    (dst / "metal").mkdir(exist_ok=True)
    for old in (base / "src").iterdir():
        if old.name != "metal" and not (dst / old.name).exists():
            (dst / old.name).symlink_to(old)
    for old in (base / "src/metal").iterdir():
        new = dst / "metal" / old.name
        if new.exists() or new.is_symlink():
            new.unlink()
        new.symlink_to(old)
    meta = json.loads((library / "identities.json").read_text())
    for name in ("runtime.mm", "app-runtime.mm", "libmetalrobo.dylib"):
        if sha(library / name) != meta["sha256"][name]:
            raise ValueError("Unexpected physical runtime " + name)
    metal(dst, templates)
    comp = json.loads(
        (body.parent.parent / "velocity-consistent/identities.json").read_text()
    )["commands"]
    c = list(comp[0])
    i = c.index("-c")
    c[i + 1] = str(dst / "matter_kernels.metal")
    c[c.index("-o") + 1] = str(out / "NumiMatter.air")
    link = list(comp[1])
    link[-3] = str(out / "NumiMatter.air")
    link[-1] = str(out / "NumiMatter.metallib")

    def execute(args, key):
        with (out / (key + ".log")).open("w") as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)

    execute(c, "metal")
    execute(link, "metallib")
    commands = meta["commands"]
    for i, name in [(2, "runtime.mm"), (3, "app-runtime.mm")]:
        (out / name).write_text(host(library / name, templates))
        args = [
            str(out / name)
            if s == str(library / name)
            else (str(out / (name + ".o")) if s == str(library / (name + ".o")) else s)
            for s in commands[i]
        ]
        args[1:1] = ["-iquote", str(library)]
        execute(args, name)
    dylib = [
        s.replace(
            str(library / "libmetalrobo.dylib"), str(out / "libmetalrobo.dylib")
        ).replace(str(library / "runtime.mm.o"), str(out / "runtime.mm.o"))
        for s in commands[4]
    ]
    execute(dylib, "dylib")
    args = [
        str(body / "numi-human-native.o")
        if s.endswith("/numi-human-native.o")
        else s.replace(
            str(library / "libmetalrobo.dylib"), str(out / "libmetalrobo.dylib")
        )
        .replace(str(library / "app-runtime.mm.o"), str(out / "app-runtime.mm.o"))
        .replace(str(library / "numi-human-native"), str(out / "numi-human-native"))
        for s in commands[5]
    ]
    execute(args, "human")
    pins = {
        k: sha(out / k)
        for k in ("numi-human-native", "libmetalrobo.dylib", "NumiMatter.metallib")
    }
    (out / "build-pins.json").write_text(json.dumps(pins, indent=2) + "\n")


if __name__ == "__main__":
    build(*(Path(x).resolve() for x in sys.argv[1:]))
