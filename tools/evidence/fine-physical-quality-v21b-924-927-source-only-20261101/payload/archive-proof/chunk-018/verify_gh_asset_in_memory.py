#!/usr/bin/env python3
"""Fetch a GH release asset into RAM, verify archive/NPZ payloads, restore one NPZ."""
import ast
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
import struct
from pathlib import Path
import datetime

import numpy as np

H = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
PLAN = H / "quality-volume-archive-chunk-018-plan/selection-plan.json"
META = H / "quality-volume-archive-chunk-018-plan/github-release-metadata.json"
OUT = H / "quality-volume-archive-chunk-018-plan"
TAG = "human-quality-volume-failed-unadmitted-20261101-chunk-018"
ASSET_NAME = "quality-volume-archive-chunk-018.tar.zst"
EXPECTED_SHA = "7c9115521e3da880c7d9c328c541a2f7ad47d0c7e8a04f51d94c662613bfc258"
EXPECTED_SIZE = 39968525
RESTORE_ROOT = Path("/Users/numi/Work/.quality-volume-archive-chunk-018-restore")
RESTORE_TARGET = Path("all-quality-847-root/round-01/proposal/fat-coupled-quality-point-proposal.npz")
LOCAL_ARCHIVE = OUT / "quality-volume-archive-chunk-018.tar.zst"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def npy_header(fp, name):
    ver = np.lib.format.read_magic(fp)
    if ver == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif ver == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    elif ver == (3, 0):
        rawlen = fp.read(4)
        if len(rawlen) != 4:
            raise RuntimeError("truncated v3 header length: " + name)
        n = struct.unpack("<I", rawlen)[0]
        if n > 10000:
            raise RuntimeError("oversized v3 header: " + name)
        raw = fp.read(n)
        if len(raw) != n:
            raise RuntimeError("truncated v3 header: " + name)
        obj = ast.literal_eval(raw.decode("utf-8").strip())
        shape, fortran, dtype = tuple(obj["shape"]), bool(obj["fortran_order"]), np.dtype(obj["descr"])
    else:
        raise RuntimeError("unsupported NPY version: " + repr(ver))
    if not isinstance(shape, tuple) or not isinstance(fortran, bool):
        raise RuntimeError("invalid NPY header: " + name)
    return {
        "name": name,
        "version": list(ver),
        "shape": list(shape),
        "fortran_order": fortran,
        "dtype": repr(dtype.descr),
        "dtype_str": dtype.str,
        "hasobject": bool(dtype.hasobject),
    }


def main():
    plan = json.loads(PLAN.read_text())
    metadata = json.loads(META.read_text())
    asset = next((a for a in metadata["assets"] if a["name"] == ASSET_NAME), None)
    if asset is None:
        raise RuntimeError("release metadata has no expected asset")
    if asset.get("digest") != "sha256:" + EXPECTED_SHA or asset.get("size") != EXPECTED_SIZE:
        raise RuntimeError("GitHub asset metadata differs from pinned local archive")
    if RESTORE_ROOT.exists():
        raise RuntimeError("refusing existing restore directory")

    cmd = [
        "gh", "release", "download", TAG, "--repo", "Numi2/numilab-human",
        "--pattern", ASSET_NAME, "--output", "-"
    ]
    remote = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if remote.returncode != 0:
        raise RuntimeError("gh release download failed: " + remote.stderr.decode("utf-8", "replace")[-1000:])
    archive_bytes = remote.stdout
    remote_sha = sha256(archive_bytes)
    if len(archive_bytes) != EXPECTED_SIZE or remote_sha != EXPECTED_SHA:
        raise RuntimeError("independent GH stream size/SHA mismatch")
    tar_result = subprocess.run(
        ["/opt/homebrew/bin/zstd", "-dc"],
        input=archive_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180
    )
    if tar_result.returncode != 0:
        raise RuntimeError("zstd decode failed: " + tar_result.stderr.decode("utf-8", "replace")[-1000:])
    tar_bytes = tar_result.stdout

    index = None
    selection_bytes = None
    blobs = []
    seen_names = set()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        for member in tf:
            if not member.isfile() or member.name in seen_names:
                raise RuntimeError("unexpected/non-file/duplicate tar member: " + member.name)
            seen_names.add(member.name)
            fp = tf.extractfile(member)
            if member.name == "archive-index.json":
                index = json.load(fp)
            elif member.name == "selection-and-retirement.json":
                selection_bytes = fp.read()
            elif member.name.startswith("blobs/") and ".." not in Path(member.name).parts:
                digest = member.name.split("/", 1)[1]
                blob = fp.read()
                if len(blob) != member.size or sha256(blob) != digest:
                    raise RuntimeError("blob digest or size mismatch: " + digest)
                blobs.append((digest, blob))
            else:
                raise RuntimeError("unexpected archive member: " + member.name)

    if index is None or selection_bytes is None:
        raise RuntimeError("archive index or selection receipt missing")
    if sha256(selection_bytes) != sha256(PLAN.read_bytes()):
        raise RuntimeError("embedded selection plan differs from local plan")
    if index.get("unique_blob_count") != 2 or len(blobs) != 2:
        raise RuntimeError("unexpected blob count")
    if sum(len(blob) for _, blob in blobs) != 66831091:
        raise RuntimeError("uncompressed blob byte total mismatch")
    expected = {row["expected_sha256_from_source_receipt"] for row in plan["files"]}
    if {digest for digest, _ in blobs} != expected:
        raise RuntimeError("archive blob set differs from selected files")

    headers_total = 0
    file_records = []
    restored = False
    restore_blob = None
    restore_digest = None
    for digest, blob in blobs:
        record = {"sha256": digest, "size_bytes": len(blob)}
        with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError("NPZ ZIP CRC failed at " + bad)
            members = [name for name in zf.namelist() if name.endswith(".npy")]
            if not members:
                raise RuntimeError("NPZ contains no NPY arrays")
            required = {"points_world_m.npy", "points_world_m_f32.npy", "tetrahedra.npy", "material_indices.npy", "boundary_faces.npy"}
            if not required.issubset({Path(name).name for name in members}):
                raise RuntimeError("NPZ is missing required volume arrays")
            headers = []
            for name in members:
                with zf.open(name, "r") as f:
                    headers.append(npy_header(f, name))
            record["zip_crc"] = "PASS"
            record["npy_header_count"] = len(headers)
            record["npy_header_summary_sha256"] = sha256(json.dumps(headers, sort_keys=True, separators=(",", ":")).encode())
            record["required_volume_arrays"] = sorted(required)
            record["source_paths"] = [row["source_relative_path"] for row in index["files"] if row["sha256"] == digest]
            headers_total += len(headers)
        if any(row["path"].endswith("fat-coupled-quality-point-proposal.npz") and row["sha256"] == digest for row in index["files"]):
            restore_blob = blob
            restore_digest = digest
            restored = True
        file_records.append(record)
    if not restored or restore_blob is None or restore_digest is None:
        raise RuntimeError("selected Fat NPZ was not identified for restore")
    if not all(r["zip_crc"] == "PASS" for r in file_records):
        raise RuntimeError("some NPZ CRC failed")
    if not LOCAL_ARCHIVE.is_file() or sha256(LOCAL_ARCHIVE.read_bytes()) != EXPECTED_SHA:
        raise RuntimeError("local archive differs before safe removal")
    LOCAL_ARCHIVE.unlink()
    target = RESTORE_ROOT / RESTORE_TARGET
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(restore_blob)
    restored_sha = sha256(target.read_bytes())
    if restored_sha != restore_digest or target.stat().st_size != len(restore_blob):
        raise RuntimeError("exact restore file mismatch")
    for record in file_records:
        if record["sha256"] == restore_digest:
            record["restore_path"] = str(target)
            record["restored_sha256"] = restored_sha
    if not all(r.get("restored_sha256") == restore_digest for r in file_records if r["sha256"] == restore_digest):
        raise RuntimeError("restored member receipt mismatch")

    canonical_index = sha256(json.dumps(index, sort_keys=True, separators=(",", ":")).encode())
    readback = {
        "schema": "numi-human.quality-volume-npz-readback.v1",
        "status": "PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS",
        "remote_gh_asset_stdout_sha256": remote_sha,
        "archive_index_canonical_sha256": canonical_index,
        "npz_count": len(file_records),
        "uncompressed_blob_bytes": sum(r["size_bytes"] for r in file_records),
        "npy_array_headers_read": headers_total,
        "all_zip_crcs_passed": True,
        "files": file_records,
        "physical_admission": False,
        "native_steps": 0,
    }
    restore_receipt = {
        "schema": "numi-human.volume-archive-one-npz-restore-receipt.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "PASS_EXACT_SELECTED_NPZ_RESTORE",
        "archive_sha256": remote_sha,
        "remote_download_method": "gh release download --output - captured in memory; no independent asset file was retained",
        "restored_original_path": str(H / RESTORE_TARGET),
        "restored_path": str(RESTORE_ROOT / RESTORE_TARGET),
        "restored_sha256": sha256((RESTORE_ROOT / RESTORE_TARGET).read_bytes()),
        "restored_size_bytes": (RESTORE_ROOT / RESTORE_TARGET).stat().st_size,
        "expected_source_sha256": "7bac333690592f27ea724317cab6303f07bfe3039f89e855b5a652ace616dfd3",
        "physical_admission": False,
        "native_steps": 0,
    }
    (OUT / "npz-readback-report.json").write_text(json.dumps(readback, indent=2, sort_keys=True) + "\n")
    (OUT / "one-npz-restore-receipt.json").write_text(json.dumps(restore_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": readback["status"],
        "asset_sha256": remote_sha,
        "blob_count": len(file_records),
        "blob_bytes": readback["uncompressed_blob_bytes"],
        "npy_headers": headers_total,
        "restore_sha256": restore_receipt["restored_sha256"],
        "restore_path": restore_receipt["restored_path"],
        "report": str(OUT / "npz-readback-report.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
