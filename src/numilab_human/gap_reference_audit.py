"""Validate the local evidence links named by the Human gap registry.

This checks whether registered documents and their local Markdown links can be
resolved and identifies the exact files inspected. It does not validate the
scientific claims inside those files or promote any Human qualification gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from . import gap_execution
from .model import ImportError as HumanImportError
from .target_coverage import canonical_bytes, digest, file_digest


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.human-gap-reference-audit.v1"
COMPILER = "numilab-human.gap-reference-audit.1"
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_LINK = re.compile(r"(?P<image>!)?\[[^\]]*\]\((?P<destination><[^>\n]*>|[^)\n]+)\)")
_LINK_OPENER = re.compile(r"\]\s*\(")
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.*?)\1(?!`)", re.DOTALL)
_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
_CUSTOM_HEADING_ID = re.compile(r"\s+\{#([^}]+)\}\s*$")
_HTML_ID = re.compile(r"<(?:a|[^>]+)\b[^>]*\bid=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


def _without_code(text: str) -> str:
    """Blank fenced and inline code while preserving line/column offsets."""
    visible: list[str] = []
    fence_char: str | None = None
    fence_size = 0
    for line in text.splitlines(keepends=True):
        match = _FENCE.match(line)
        if fence_char is None and match:
            fence_char, fence_size = match.group(1)[0], len(match.group(1))
            visible.append("\n" if line.endswith("\n") else "")
            continue
        if fence_char is not None:
            if match and match.group(1)[0] == fence_char and len(match.group(1)) >= fence_size:
                fence_char, fence_size = None, 0
            visible.append("\n" if line.endswith("\n") else "")
            continue
        visible.append(line)
    text = "".join(visible)

    def blank(match: re.Match[str]) -> str:
        value = match.group(0)
        return "".join("\n" if char == "\n" else " " for char in value)

    return _INLINE_CODE.sub(blank, text)


def _heading_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        explicit = _CUSTOM_HEADING_ID.search(line)
        if explicit:
            anchors.add(explicit.group(1))
            line = line[:explicit.start()]
        heading = _HEADING.match(line)
        if not heading:
            continue
        title = html.unescape(heading.group(1))
        title = re.sub(r"`([^`]*)`", r"\1", title)
        title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
        title = re.sub(r"<[^>]*>", "", title).lower()
        slug = re.sub(r"[^\w\- ]", "", title, flags=re.UNICODE).strip()
        # GitHub anchors replace each whitespace character separately; notably,
        # a heading with "title — date" retains the double hyphen around the
        # stripped em dash in its generated fragment.
        slug = re.sub(r"\s", "-", slug)
        ordinal = counts.get(slug, 0)
        counts[slug] = ordinal + 1
        anchors.add(slug if ordinal == 0 else f"{slug}-{ordinal}")
    anchors.update(_HTML_ID.findall(text))
    return anchors


def _relative_path(root: Path, source: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return source.resolve()
    decoded = unquote(raw_path).replace(r"\ ", " ")
    candidate = Path(decoded)
    if candidate.is_absolute():
        return None
    resolved = (source.parent / candidate).resolve()
    if not resolved.is_relative_to(root):
        return None
    return resolved


def _links_in_markdown(path: Path, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(root).resolve()
    path = Path(path).resolve()
    raw_text = path.read_text(encoding="utf-8")
    text = _without_code(raw_text)
    anchors = _heading_anchors(raw_text)
    links: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    for match in _LINK.finditer(text):
        destination = match.group("destination").strip()
        if destination.startswith("<") and destination.endswith(">"):
            target = destination[1:-1]
        else:
            target = destination.split(None, 1)[0] if destination else ""
        target = re.sub(r"\\([\\`*{}\[\]()#+.!_|<>-])", r"\1", target)
        parsed = urlsplit(target)
        line = text.count("\n", 0, match.start()) + 1
        record = {
            "source": path.relative_to(root).as_posix(),
            "line": line,
            "kind": "image" if match.group("image") else "link",
            "target": target,
        }
        if parsed.scheme or parsed.netloc:
            external.append({**record, "scheme": parsed.scheme or "network-path"})
            continue
        local = _relative_path(root, path, parsed.path)
        if local is None or not local.exists():
            links.append({**record, "status": "missing_or_outside_repository"})
            continue
        if not local.is_file() and not local.is_dir():
            links.append({**record, "status": "not_a_file_or_directory"})
            continue
        if parsed.fragment and local.is_file() and local.suffix.lower() in {".md", ".markdown"}:
            try:
                target_anchors = _heading_anchors(local.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                target_anchors = set()
            if unquote(parsed.fragment) not in target_anchors:
                links.append({**record, "status": "missing_anchor"})
                continue
        elif parsed.fragment and not parsed.path and unquote(parsed.fragment) not in anchors:
            links.append({**record, "status": "missing_anchor"})
            continue
        links.append({**record, "status": "resolved", "path": local.relative_to(root).as_posix(),
                      "target_type": "directory" if local.is_dir() else "file"})
    return links, external


def _predicate_hashes() -> dict[str, str]:
    return {
        "gap_reference_audit.py": file_digest(Path(__file__).resolve()),
        "gap_execution.py": file_digest(Path(gap_execution.__file__).resolve()),
        "target_coverage.py": file_digest(Path(__file__).with_name("target_coverage.py")),
    }


def _json_parse(path: Path) -> None:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            json.load(stream)
    elif path.suffix.lower() == ".json":
        json.loads(path.read_text(encoding="utf-8"))


def materialize(registry: dict[str, Any], *, root: Path = ROOT,
                registry_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    registry_path = (root / "config/human-gap-execution.v1.json" if registry_path is None
                     else Path(registry_path).resolve())
    if not registry_path.is_relative_to(root) or not registry_path.is_file():
        raise HumanImportError("gap registry must be a retained file inside the repository")
    direct_references = sorted({registry["ledger"], *(ref for stream in registry["workstreams"]
                                                      for ref in stream["references"])})
    gap_execution.validate_registry(registry, root=root)
    registry_hash = digest(registry)
    registry_file_hash = file_digest(registry_path) if registry_path.is_file() else None
    direct_hashes: dict[str, str] = {}
    json_validated: list[str] = []
    markdown: list[Path] = []
    predicate_hashes = _predicate_hashes()
    for relative in direct_references:
        path = gap_execution._path(root, relative)
        direct_hashes[relative] = file_digest(path)
        if path.suffix.lower() in {".md", ".markdown"}:
            markdown.append(path)
        if path.name.endswith(".json.gz") or path.suffix.lower() == ".json":
            try:
                _json_parse(path)
            except (OSError, UnicodeError, ValueError) as error:
                raise HumanImportError(f"invalid JSON evidence reference: {relative}") from error
            json_validated.append(relative)

    link_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    unparsed_link_openers: list[dict[str, Any]] = []
    for path in markdown:
        links, external = _links_in_markdown(path, root)
        link_rows.extend(links)
        external_rows.extend(external)
        text = _without_code(path.read_text(encoding="utf-8"))
        opener_count = len(_LINK_OPENER.findall(text))
        parsed_count = len(links) + len(external)
        if opener_count != parsed_count:
            unparsed_link_openers.append({
                "source": path.relative_to(root).as_posix(),
                "markdown_link_opener_count": opener_count,
                "parsed_link_count": parsed_count,
            })
    unresolved = [row for row in link_rows if row["status"] != "resolved"]

    linked_file_hashes: dict[str, dict[str, Any]] = {}
    directory_rows: dict[str, dict[str, Any]] = {}
    for row in link_rows:
        if row["status"] != "resolved":
            continue
        target_path = root / row["path"]
        if row["target_type"] == "file":
            linked_file_hashes[row["path"]] = {
                "bytes": target_path.stat().st_size,
                "sha256": file_digest(target_path),
            }
        else:
            names = sorted(entry.name + ("/" if entry.is_dir() else "")
                           for entry in target_path.iterdir())
            directory_rows[row["path"]] = {
                "immediate_entry_count": len(names),
                "immediate_entries_sha256": hashlib.sha256(
                    canonical_bytes(names)).hexdigest(),
            }

    # Fail closed if any inspected file or directory changed during the audit.
    for relative, expected in direct_hashes.items():
        if file_digest(root / relative) != expected:
            raise HumanImportError(f"evidence reference changed during audit: {relative}")
    for relative, identity in linked_file_hashes.items():
        if file_digest(root / relative) != identity["sha256"]:
            raise HumanImportError(f"linked evidence changed during audit: {relative}")
    for relative, identity in directory_rows.items():
        path = root / relative
        names = sorted(entry.name + ("/" if entry.is_dir() else "") for entry in path.iterdir())
        if (len(names) != identity["immediate_entry_count"]
                or hashlib.sha256(canonical_bytes(names)).hexdigest() !=
                identity["immediate_entries_sha256"]):
            raise HumanImportError(f"linked evidence directory changed during audit: {relative}")
    if registry_file_hash is not None and file_digest(registry_path) != registry_file_hash:
        raise HumanImportError("gap registry changed during audit")
    if _predicate_hashes() != predicate_hashes:
        raise HumanImportError("evidence reference audit source changed during scan")

    domains = sorted({urlsplit(row["target"]).netloc for row in external_rows
                      if urlsplit(row["target"]).netloc})
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "registry_sha256": registry_hash,
        "registry_file_sha256": registry_file_hash,
        "predicate_source_sha256": predicate_hashes,
        "registered_reference_sha256": direct_hashes,
        "validated_json_references": sorted(json_validated),
        "local_markdown_targets": link_rows,
        "linked_file_sha256": linked_file_hashes,
        "linked_directories": directory_rows,
        "external_links_not_fetched": external_rows,
        "external_domains_not_verified": domains,
        "unresolved_local_targets": unresolved,
        "unparsed_markdown_link_syntax": unparsed_link_openers,
        "status": ("passed_local_reference_graph"
                   if not unresolved and not unparsed_link_openers
                   else "failed_local_reference_graph"),
        "counts": {
            "registered_references": len(direct_references),
            "registered_markdown_documents": len(markdown),
            "json_references_parsed": len(json_validated),
            "local_markdown_links_and_images": len(link_rows),
            "resolved_local_file_targets": len(linked_file_hashes),
            "resolved_local_directory_targets": len(directory_rows),
            "unresolved_local_targets": len(unresolved),
            "unparsed_markdown_link_syntax": len(unparsed_link_openers),
            "external_links_not_fetched": len(external_rows),
            "external_domains_not_verified": len(domains),
        },
        "evidence_boundary": (
            "This audit verifies registered local reference availability, Markdown inline "
            "link/image paths and local Markdown heading anchors, and JSON syntax. It does "
            "not fetch external URLs, validate source claims or embedded scientific receipts, "
            "assess task readiness, or qualify Human anatomy, physiology or runtime behavior."
        ),
    }
    result["audit_sha256"] = digest(result)
    return result


def _write_immutable(path: Path, encoded: bytes) -> None:
    path = Path(path)
    if path.is_symlink():
        raise HumanImportError("audit output is redirected by a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise HumanImportError("audit output is immutable; choose a new path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError as error:
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise HumanImportError("audit output changed during publication") from error


def command(arguments: argparse.Namespace) -> int:
    registry_path = Path(arguments.registry).resolve()
    registry = gap_execution.read_json(registry_path)
    result = materialize(registry, root=arguments.repository_root,
                         registry_path=registry_path)
    encoded = canonical_bytes(result) + b"\n"
    if arguments.output is None:
        print(encoded.decode("utf-8"), end="")
    else:
        _write_immutable(arguments.output, encoded)
        print(json.dumps({"audit_sha256": result["audit_sha256"], **result["counts"],
                          "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "passed_local_reference_graph" else 2


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=ROOT / "config/human-gap-execution.v1.json")
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="new immutable audit receipt; default is stdout")
    parser.set_defaults(handler=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except (HumanImportError, OSError) as error:
        parser.exit(2, f"gap-reference-audit: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
