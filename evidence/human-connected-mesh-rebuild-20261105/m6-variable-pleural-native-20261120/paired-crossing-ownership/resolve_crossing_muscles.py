"""Resolve audited rendered muscle bank IDs through the source payload manifest."""
import hashlib
import json
from pathlib import Path
import sys

launch, trial = map(Path,sys.argv[1:])
arguments = json.loads(launch.read_text())["argv"]
payload = Path(arguments[arguments.index("--soft-tissue-payload")+1])
manifest = payload.with_suffix(".manifest.json")
composer = payload.with_name("report.json")

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

composition = json.loads(composer.read_text())
payload_digest, manifest_digest = sha(payload), sha(manifest)
assert composition["payload_sha256"] == payload_digest
assert composition["manifest_sha256"] == manifest_digest
source = json.loads(manifest.read_text())
assert source["payload"]["sha256"] == payload_digest
entries = source["source"]["surfaces"]
labels = {row["stable_id"]:row for row in entries}
assert len(labels)==len(entries)
typed = json.loads((trial/"crossings-by-physical-scope.json").read_text())
target_counts = {row["stable_id"]:row["target_faces"] for row in typed["surfaces"]
                 if row["semantic"]==51005}
comparison = json.loads((trial/"complete-paired-pairs.json").read_text())
muscle = []
for row in comparison["banks"]:
    semantic, stable = row["key"]
    if semantic != 51005 or not row["initial"] and not row["terminal"]:
        continue
    entry = labels[stable]
    muscle.append(dict(stable_id=stable,label=entry.get("label") or entry["source_name"],initial=row["initial"],
                       terminal=row["terminal"],appeared=row["appeared"],cleared=row["cleared"],
                       native_faces=target_counts[stable], manifest_faces=entry["triangle_count"]))
muscle.sort(key=lambda x:(-x["initial"],x["stable_id"]))
print(json.dumps(dict(
    scope="Manifest names for the soft-tissue CLI path used by the matched native replay",
    payload_sha256=payload_digest,manifest_sha256=manifest_digest,
    authored_manifest_report_sha256=sha(composer),surfaces=muscle),indent=2))
