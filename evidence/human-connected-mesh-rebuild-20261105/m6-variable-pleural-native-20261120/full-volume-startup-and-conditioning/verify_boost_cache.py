"""Retire only the byte-verified extracted copy of the pinned Boost distribution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile

cache = Path.home() / "Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/authoring-deps/cgal-autorefine-540"
archive = cache / "download/boost_1_87_0.tar.bz2"
receipt = json.loads(archive.with_suffix(".bz2.json").read_text())
source = cache / "source/boost_1_87_0"
assert source.is_dir() and not source.is_symlink() and source.resolve() == source
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

assert digest(archive) == receipt["sha256"] and receipt["file"] == archive.name
assert not any(re.search(r"(clang|cmake|ninja|tar).*" + str(source), line)
               for line in subprocess.check_output(["ps", "-Ao", "args"], text=True).splitlines())
matches = set()
logical_bytes = 0
allocated = 0
with tarfile.open(archive, "r|bz2") as bundle:
    for member in bundle:
        relative = Path(member.name)
        assert relative.parts[0] == source.name and ".." not in relative.parts
        local = source.parent / relative
        if local in matches:
            raise ValueError(f"Duplicate archive path {member.name}")
        if local.is_symlink():
            raise ValueError(f"Unexpected symlink {local}")
        matches.add(local)
        if member.isdir():
            assert local.is_dir(), local
            continue
        assert member.isreg() and local.is_file(), local
        assert local.stat().st_size == member.size, local
        assert stat.S_IMODE(local.stat().st_mode) == member.mode, local
        with bundle.extractfile(member) as data:
            assert hashlib.file_digest(data, "sha256").hexdigest() == digest(local), local
        logical_bytes += member.size
        allocated += local.stat().st_blocks * 512
existing = set(source.rglob("*")) | {source}
assert existing == matches, list(sorted(existing ^ matches))[:20]
# Metadata from a local build would make this a modified tree.
attrs = subprocess.check_output(["/usr/bin/xattr", "-r", str(source)], text=True)
assert not attrs.strip(), attrs[:300]
retired = source.with_name(source.name + ".verified-extract-retirement")
assert not retired.exists()
assert digest(archive) == receipt["sha256"]
result = dict(archive=str(archive), archive_sha256=receipt["sha256"], source=str(source),
              verified_entries=len(matches), logical_bytes=logical_bytes,
              allocated_file_bytes=allocated, restoration=["tar","-xjf",str(archive),"-C",str(source.parent)])
path = Path(__file__).with_name("boost-cache-retirement.json")
path.write_text(json.dumps(result, indent=2) + "\n")
source.rename(retired)
shutil.rmtree(retired)
print(json.dumps(result), flush=True)
