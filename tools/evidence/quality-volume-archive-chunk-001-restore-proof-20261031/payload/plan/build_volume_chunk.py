#!/usr/bin/env python3
"""Build one receipt-pinned historical volume archive chunk.

No source is removed or modified. The caller must pass --build to hash the
selected payloads and create the streamed content-addressed tar.zst.
"""
import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_has_pin(receipt_path, target_path, expected_sha):
    target = str(Path(target_path).resolve())
    data = json.loads(Path(receipt_path).read_text())
    found = []

    def visit(node):
        if isinstance(node, dict):
            path_value = node.get("path")
            sha_value = node.get("sha256")
            if isinstance(path_value, str):
                candidate = Path(path_value)
                if candidate.is_absolute() and str(candidate.resolve()) == target:
                    found.append(sha_value)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(data)
    return expected_sha in found


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def read(self, size=-1):
        block = self.stream.read(size)
        if block:
            self.digest.update(block)
            self.byte_count += len(block)
        return block


def add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def prepare(plan_path):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get("schema") != "numi-human.quality-volume-archive-chunk-plan.v1":
        raise RuntimeError("unexpected selection plan schema")
    if plan.get("preparation_status") != "READY_FOR_REVIEW_LIVE_PAYLOAD_HASH_PENDING":
        raise RuntimeError("selection plan is not in the prepared state")
    root = Path(plan["source_root"]).resolve()
    rows = plan["files"]
    if len(rows) != plan["chunk"]["file_count"]:
        raise RuntimeError("file count differs from plan")
    if sum(row["size_bytes_from_stat"] for row in rows) != plan["chunk"]["original_bytes"]:
        raise RuntimeError("byte total differs from plan")
    if len({row["path"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate selected paths")

    checked = []
    for row in rows:
        source = Path(row["path"])
        if not source.is_absolute() or source.resolve() != source:
            raise RuntimeError("source path must be absolute and canonical: " + str(source))
        if source.is_symlink() or not source.is_file():
            raise RuntimeError("selected source is missing, non-regular, or symlinked: " + str(source))
        if source.stat().st_size != row["size_bytes_from_stat"]:
            raise RuntimeError("selected source size changed: " + str(source))
        expected = row["expected_sha256_from_source_receipt"]
        if len(expected) != 64:
            raise RuntimeError("invalid receipt SHA: " + str(source))
        if not receipt_has_pin(row["source_receipt_path"], source, expected):
            raise RuntimeError("source receipt no longer contains the selected path/SHA: " + str(source))
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError("live file SHA differs from source receipt: " + str(source))
        relative = source.relative_to(root)
        if ".." in relative.parts:
            raise RuntimeError("source path escapes archive root: " + str(source))
        checked.append((row, source, relative.as_posix(), actual))

    return plan, checked


def build(plan, checked, archive_path):
    archive = Path(archive_path)
    if archive.exists():
        raise RuntimeError("refusing to replace existing archive: " + str(archive))
    partial = archive.with_name(archive.name + ".partial")
    if partial.exists():
        raise RuntimeError("refusing to replace existing partial archive: " + str(partial))
    archive.parent.mkdir(parents=True, exist_ok=True)

    files = []
    unique = {}
    for row, source, relative, actual in checked:
        out = {
            "path": str(source),
            "source_relative_path": relative,
            "sha256": actual,
            "size_bytes": row["size_bytes_from_stat"],
            "kind": "historical_volume_proposal",
            "archive_member": "blobs/" + actual,
        }
        files.append(out)
        unique.setdefault(actual, {"path": str(source), "size_bytes": row["size_bytes_from_stat"]})

    index = {
        "schema": "numi-human.quality-volume-proposal-archive.v1",
        "snapshot": {
            "name": plan["chunk"]["id"],
            "not_current_input": True,
            "source_workflow_commit": plan["source_workflow"]["workflow_commit"],
        },
        "selection_plan_path": str(Path(args.plan).resolve()),
        "files": files,
        "unique_blobs": [
            {"archive_member": "blobs/" + digest, "sha256": digest, "size_bytes": blob["size_bytes"]}
            for digest, blob in sorted(unique.items())
        ],
        "unique_blob_count": len(unique),
        "unique_blob_bytes": sum(blob["size_bytes"] for blob in unique.values()),
        "file_reference_count": len(files),
    }
    index_path = Path(args.plan).with_name("archive-index.json")
    index_bytes = (json.dumps(index, indent=2, sort_keys=True) + "\n").encode()
    selection_bytes = Path(args.plan).read_bytes()
    zstd = Path("/opt/homebrew/bin/zstd")
    if not zstd.is_file():
        raise RuntimeError("pinned zstd executable is missing: " + str(zstd))

    with partial.open("xb") as archive_stream:
        process = subprocess.Popen([str(zstd), "-T1", "-6", "-c"], stdin=subprocess.PIPE, stdout=archive_stream)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|") as tar:
                add_bytes(tar, "archive-index.json", index_bytes)
                add_bytes(tar, "selection-and-retirement.json", selection_bytes)
                emitted = set()
                for row, source, relative, digest in checked:
                    if digest in emitted:
                        continue
                    emitted.add(digest)
                    info = tarfile.TarInfo("blobs/" + digest)
                    info.size = source.stat().st_size
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with source.open("rb") as stream:
                        reader = HashingReader(stream)
                        tar.addfile(info, reader)
                    if reader.digest.hexdigest() != digest or reader.byte_count != info.size:
                        raise RuntimeError("source changed while archiving: " + str(source))
            process.stdin.close()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError("zstd exited with status " + str(return_code))
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise

    os.replace(partial, archive)
    index_path.write_bytes(index_bytes)
    receipt = {
        "status": "PASS_LOCAL_STREAMED_ARCHIVE_BUILD",
        "archive_path": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "archive_size_bytes": archive.stat().st_size,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "selection_plan_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "selected_file_count": len(files),
        "unique_blob_count": len(unique),
        "original_bytes": sum(item["size_bytes"] for item in files),
        "independent_release_download_verified": False,
        "source_retirement_performed": False,
    }
    (Path(args.plan).with_name("archive-build-receipt.json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--build", action="store_true")
    global args
    args = parser.parse_args()
    if not args.build:
        plan = json.loads(Path(args.plan).read_text())
        print(json.dumps({
            "status": "PREPARED_NO_PAYLOAD_HASHING",
            "chunk": plan["chunk"]["id"],
            "file_count": plan["chunk"]["file_count"],
            "original_bytes": plan["chunk"]["original_bytes"],
            "build_requires_explicit_flag": "--build",
        }, sort_keys=True))
        return
    plan, checked = prepare(args.plan)
    build(plan, checked, args.archive)


if __name__ == "__main__":
    main()
