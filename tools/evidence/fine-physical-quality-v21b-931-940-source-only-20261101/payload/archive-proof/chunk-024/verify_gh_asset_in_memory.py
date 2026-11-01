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
PLAN = H / "quality-volume-archive-chunk-024-plan/selection-plan.json"
META = H / "quality-volume-archive-chunk-024-plan/github-release-metadata.json"
OUT = H / "quality-volume-archive-chunk-024-plan"
TAG = "human-quality-volume-failed-unadmitted-20261101-chunk-024"
ASSET_NAME = "quality-volume-archive-chunk-024.tar.zst"
EXPECTED_SHA = "54f7301b61d030923a4ca74924b96b95b2ea6097e5b2998cbc73ef4908aec767"
EXPECTED_SIZE = 39936085
RESTORE_TARGET = Path("all-quality-873-root/round-01/proposal/fat-coupled-quality-point-proposal.npz")
LOCAL_ARCHIVE = OUT / "quality-volume-archive-chunk-024.tar.zst"


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
    if metadata.get("tagName") != TAG or asset.get("id") != "RA_kwDOUElluM4pjQi6":
        raise RuntimeError("GitHub release tag or asset identity differs from pinned chunk 024")
    if asset.get("digest") != "sha256:" + EXPECTED_SHA or asset.get("size") != EXPECTED_SIZE:
        raise RuntimeError("GitHub asset metadata differs from pinned local archive")
    # Restore is intentionally performed in RAM; no filesystem restore path is used.

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
    if sum(len(blob) for _, blob in blobs) != 66831229:
        raise RuntimeError("uncompressed blob byte total mismatch")
    expected = {row["expected_sha256_from_source_receipt"] for row in plan["files"]}
    if {digest for digest, _ in blobs} != expected:
        raise RuntimeError("archive blob set differs from selected files")
    expected_records = {row["path"]: row["expected_sha256_from_source_receipt"] for row in plan["files"]}
    index_records = {row["path"]: row["sha256"] for row in index["files"]}
    if index_records != expected_records:
        raise RuntimeError("archive index path/hash mapping differs from selection plan")

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
    # Rehydrate the selected NPZ into a fresh in-memory byte stream, then hash,
    # verify ZIP CRC, read NPY headers and load each required native array.
    restore_buffer = io.BytesIO()
    restore_buffer.write(restore_blob)
    restore_buffer.seek(0)
    restored_bytes = restore_buffer.read()
    restored_sha = sha256(restored_bytes)
    if restored_sha != restore_digest or len(restored_bytes) != len(restore_blob):
        raise RuntimeError("exact in-memory restore hash/size mismatch")
    restored_zip_crc = "PASS"
    restored_headers = []
    with zipfile.ZipFile(io.BytesIO(restored_bytes), "r") as restored_npz_zip:
        bad = restored_npz_zip.testzip()
        if bad is not None:
            raise RuntimeError("restored NPZ ZIP CRC failed at " + bad)
        for member in restored_npz_zip.namelist():
            if member.endswith(".npy"):
                with restored_npz_zip.open(member, "r") as f:
                    restored_headers.append(npy_header(f, member))
    required_restore_arrays = ("points_world_m", "points_world_m_f32", "tetrahedra", "material_indices", "boundary_faces")
    restored_arrays = {}
    with np.load(io.BytesIO(restored_bytes), allow_pickle=False) as restored_npz:
        if not set(required_restore_arrays).issubset(set(restored_npz.files)):
            raise RuntimeError("restored in-memory NPZ lacks required volume arrays")
        for name in required_restore_arrays:
            arr = restored_npz[name]
            restored_arrays[name] = {
                "shape": list(arr.shape),
                "dtype": arr.dtype.str,
                "array_bytes_sha256": sha256(arr.tobytes(order="C")),
            }
            del arr
    restored_header_summary_sha256 = sha256(json.dumps(restored_headers, sort_keys=True, separators=(",", ":")).encode())
    restored_zip_crc = "PASS"
    # The GH asset and exact in-memory restore are now verified; discard only the
    # task-owned local archive copy to release space.
    LOCAL_ARCHIVE.unlink()
    for record in file_records:
        if record["sha256"] == restore_digest:
            record["ram_restore_sha256"] = restored_sha
            record["ram_restore_npy_header_count"] = len(restored_headers)
            record["ram_restore_npy_header_summary_sha256"] = restored_header_summary_sha256
            record["ram_restore_required_array_readback"] = restored_arrays
    if not all(r.get("ram_restore_sha256") == restore_digest for r in file_records if r["sha256"] == restore_digest):
        raise RuntimeError("RAM restored-member receipt mismatch")

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
        "status": "PASS_EXACT_IN_MEMORY_NPZ_RESTORE_AND_READBACK",
        "archive_sha256": remote_sha,
        "remote_download_method": "gh release download --output - captured in RAM; full blob SHA verified",
        "restored_original_path": str(H / RESTORE_TARGET),
        "restored_path": None,
        "restore_method": "fresh BytesIO copy, full SHA/size check, ZIP CRC, NPY headers, and required array loads",
        "filesystem_restore_performed": False,
        "restored_zip_crc": restored_zip_crc,
        "restored_npy_header_count": len(restored_headers),
        "restored_npy_header_summary_sha256": restored_header_summary_sha256,
        "restored_required_array_readback": restored_arrays,
        "restored_sha256": restored_sha,
        "restored_size_bytes": len(restored_bytes),
        "expected_source_sha256": "f1cd4a8036236471aff5a48b5b15b7f10ad80c72ba3efec98bd931cdf2d97519",
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
