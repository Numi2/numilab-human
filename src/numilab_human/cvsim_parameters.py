"""Read CVSim21's pinned initial_ptr/mapping_ptr without executing C.

This is a source-specific, bounded assignment grammar, not a C interpreter.
Values remain in the source units. The caller owns the source lock and must
provide its expected SHA256. No physiology equations or integration live here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import struct
from typing import Any

from .model import ImportError as HumanImportError

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_SIGNATURE = "Hemo*hemo,Cardiac*cardiac,Micro_r*micro_r,System_parameters*system,Reflex*reflex,Timing*timing"
_LABELS = ("nominal", "sd", "lower", "upper")


def _require(ok: Any, message: str) -> None:
    if not ok:
        raise HumanImportError(f"CVSim parameter source: {message}")


def _mask_comments(text: str) -> str:
    """Keep all byte/line positions; quotes cannot open a C comment."""
    result = list(text)
    i = 0
    while i < len(text):
        if text[i] in "\"'":
            quote = text[i]
            i += 1
            while i < len(text) and text[i] != quote:
                _require(text[i] != "\n", "unterminated quoted token")
                i += 2 if text[i] == "\\" else 1
            _require(i < len(text), "unterminated quoted token")
            i += 1
        elif text.startswith("//", i) or text.startswith("/*", i):
            if text.startswith("//", i):
                end = text.find("\n", i)
                end = len(text) if end < 0 else end
            else:
                end = text.find("*/", i + 2)
                _require(end >= 0, "unterminated block comment")
                end += 2
            for index in range(i, end):
                if result[index] != "\n":
                    result[index] = " "
            i = end
        else:
            i += 1
    return "".join(result)


def _function(text: str, name: str) -> tuple[str, int]:
    pattern = re.compile(r"\bvoid\s+" + name + r"\s*\(([^{};]*)\)\s*\{")
    matches = list(pattern.finditer(text))
    _require(len(matches) == 1, f"expected exactly one {name} definition")
    match = matches[0]
    signature = re.sub(r"\s+", "", match[1])
    expected = _SIGNATURE + (",Parameter_vector*tmp" if name == "mapping_ptr" else "")
    _require(signature == expected, f"unsupported {name} signature")
    start = match.end()
    end = text.find("}", start)
    _require(end >= 0 and "{" not in text[start:end], f"nested/unterminated {name} body")
    return text[start:end], start


def _member(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\(\*(\w+)\)", r"\1", text).replace("->", ".")
    _require(re.fullmatch(r"(?:hemo\[[0-9]+\]|cardiac\[[0-9]+\]|reflex\[[0-9]+\]|micro_r|system|timing)\.\w+\[[0-9]+\]\[[0-9]+\]", text),
             f"unsupported source member: {text[:100]}")
    return text


def _expected_members() -> set[str]:
    """Exactly the initialized tables, not the uninitialized hemo.v[1] row."""
    layouts = {
        "hemo": (17, {"c": 1, "v": 1, "r": 1, "h": 1}),
        "cardiac": (2, {"c_sys": 2, "c_dias": 2, "v": 2, "r": 1}),
        "reflex": (2, {"set": 2, "rr": 2, "res": 4, "vt": 4, "c": 2}),
        "micro_r": (None, {"r": 5}),
        "system": (None, {"bv": 1, "hr": 1, "pth": 1, "h": 1, "w": 1, "bsa": 1, "T": 3}),
        "timing": (None, {"para": 3, "beta": 3, "alpha_r": 3, "alpha_v": 3, "alpha_cpr": 3, "alpha_cpv": 3}),
    }
    return {
        f"{owner}{'' if count is None else '[' + str(index) + ']'}.{field}[{row}][{column}]"
        for owner, (count, fields) in layouts.items()
        for index in range(count or 1)
        for field, rows in fields.items()
        for row in range(rows)
        for column in range(4)
    }


def _product(expression: str, constants: dict, members: dict, *, allow_members: bool) -> tuple[float, list[str], list[str]]:
    # Canonicalize pointer/member syntax before splitting multiplication tokens.
    compact = re.sub(r"\s+", "", expression)
    compact = re.sub(r"\(\*(\w+)\)", r"\1", compact).replace("->", ".")
    factors = compact.split("*")
    _require(1 <= len(factors) <= 8, "unsupported product length")
    # Actual numeric assignments use unsuffixed floating literals. Reject C
    # integer/octal syntax and products of local floats alone rather than
    # accidentally give them Python's different promotion/overflow semantics.
    _require(len(factors) == 1 or not (factors[0] in constants and factors[1] in constants),
             "unsupported product of local float constants")
    value = 1.0
    member_refs, constant_refs = [], []
    for factor in factors:
        if _NUMBER.fullmatch(factor):
            _require("." in factor or "e" in factor.lower(), "only source floating-point literals are supported")
            number = float(factor)
        elif factor in constants:
            number = constants[factor]["value"]
            constant_refs.append(factor)
        else:
            _require(allow_members, f"unsupported numeric expression: {expression[:100]}")
            key = _member(factor)
            _require(key in members, f"unresolved source member: {key}")
            number = members[key]["value"]
            member_refs.append(key)
        _require(math.isfinite(number), "nonfinite numeric literal")
        value *= number
        _require(math.isfinite(value), "nonfinite product")
    return value, member_refs, constant_refs


def _statements(text: str, body: str, offset: int) -> list[tuple[str, int]]:
    result = []
    start = 0
    for match in re.finditer(";", body):
        raw = body[start:match.start()]
        stripped = raw.strip()
        _require(bool(stripped), "empty statement")
        first = offset + start + len(raw) - len(raw.lstrip())
        result.append((stripped, text.count("\n", 0, first) + 1))
        start = match.end()
    _require(not body[start:].strip(), "unconsumed statement or missing semicolon")
    return result


def parse_cvsim_parameters(source: bytes | Path, *, expected_sha256: str) -> dict:
    """Return all 153 mapped values and 142 source table quadruples.

    SHA256 covers exact source bytes (including CRLF/comments). ``source_file``
    is the stable source-relative owner path for either input form. Source SD
    and range entries are retained verbatim; contradictory metadata is reported,
    not repaired. C ``float alpha/beta`` declarations are rounded to binary32
    before promotion into the source double-valued products.
    """
    _require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256), "expected SHA256 must be explicit")
    if isinstance(source, Path):
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise HumanImportError("CVSim parameter source: cannot read source path") from error
    else:
        _require(isinstance(source, bytes), "source must be bytes or Path")
        raw = source
    _require(len(raw) <= 2_000_000, "source exceeds bounded parser size")
    actual = hashlib.sha256(raw).hexdigest()
    _require(actual == expected_sha256, "source SHA256 mismatch")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HumanImportError("CVSim parameter source: invalid UTF-8") from error
    _require("\0" not in text, "NUL byte in source")
    text = text.replace("\r\n", "\n")
    masked = _mask_comments(text)
    source_file = "main/initial.c"
    constants: dict[str, dict] = {}
    members: dict[str, dict] = {}
    expected_members = _expected_members()
    body, offset = _function(masked, "initial_ptr")
    for statement, line in _statements(masked, body, offset):
        declaration = re.fullmatch(r"float\s+(alpha|beta)\s*=\s*(.+)", statement, re.DOTALL)
        if declaration:
            name, expression = declaration.groups()
            _require(name not in constants, f"duplicate constant: {name}")
            _require(_NUMBER.fullmatch(expression.strip()), "constant requires one numeric literal")
            number, _, _ = _product(expression, {}, {}, allow_members=False)
            try:
                number = struct.unpack("f", struct.pack("f", number))[0]
            except (OverflowError, struct.error) as error:
                raise HumanImportError("CVSim parameter source: nonfinite float constant") from error
            _require(math.isfinite(number), "nonfinite float constant")
            constants[name] = {"value": number, "source_expression": expression.strip(), "file": source_file, "line": line, "c_type": "float"}
            continue
        assignment = re.fullmatch(r"([^=]+)=([^=]+)", statement, re.DOTALL)
        _require(assignment, f"unconsumed scalar assignment at line {line}")
        key = _member(assignment[1])
        _require(key in expected_members, f"unsupported member/index: {key}")
        _require(key not in members, f"duplicate source member: {key}")
        expression = assignment[2].strip()
        value, _, constant_refs = _product(expression, constants, {}, allow_members=False)
        members[key] = {"value": value, "source_expression": expression, "constant_refs": constant_refs, "file": source_file, "line": line}
    _require(set(constants) == {"alpha", "beta"}, "missing alpha/beta declarations")
    _require(set(members) == expected_members, f"missing source table members: {sorted(expected_members - members.keys())[:4]}")

    parameters: dict[int, dict] = {}
    body, offset = _function(masked, "mapping_ptr")
    for statement, line in _statements(masked, body, offset):
        assignment = re.fullmatch(r"tmp\s*->\s*vec\s*\[\s*([0-9]+)\s*\]\s*=\s*([^=]+)", statement, re.DOTALL)
        _require(assignment, f"unconsumed mapping assignment at line {line}")
        index = int(assignment[1])
        _require(0 <= index < 153, f"parameter index outside source vector: {index}")
        _require(index not in parameters, f"duplicate parameter index: {index}")
        expression = assignment[2].strip()
        value, refs, _ = _product(expression, {}, members, allow_members=True)
        raw_line = text.splitlines()[line - 1]
        source_comment = raw_line.partition("//")[2].strip()
        parameters[index] = {"index": index, "value": value, "source_expression": expression, "source_members": refs,
                             "source_member": refs[0] if len(refs) == 1 else None, "file": source_file, "mapping_line": line,
                             "source_comment": source_comment, "source_member_provenance": [{"member": key, **members[key]} for key in refs]}
    _require(set(parameters) == set(range(153)), "missing parameter-vector assignments")
    referenced = {key for row in parameters.values() for key in row["source_members"]}
    tables, inconsistencies = {}, []
    for key in sorted({key.rsplit("[", 1)[0] for key in members}):
        values = {label: members[f"{key}[{column}]"]["value"] for column, label in enumerate(_LABELS)}
        provenance = {label: members[f"{key}[{column}]"] for column, label in enumerate(_LABELS)}
        tables[key] = {**values, "provenance": provenance,
                       "mapped_parameter_indices": [index for index, row in sorted(parameters.items()) if any(ref.startswith(key + "[") for ref in row["source_members"])]}
        if not values["lower"] <= values["nominal"] <= values["upper"]:
            inconsistencies.append({"id": "nominal_outside_source_range", "member": key, **values,
                                    "source_lines": sorted({row["line"] for row in provenance.values()}),
                                    "policy": "preserve_exact_source_values"})
        if values["sd"] < 0 or values["lower"] > values["upper"]:
            inconsistencies.append({"id": "invalid_source_dispersion_or_range", "member": key, **values,
                                    "policy": "preserve_exact_source_values"})
    return {"schema": "NumiHuman.CVSim21-source-parameters.v1", "source_file": source_file, "source_sha256": actual,
            "source_bytes": len(raw), "source_units_preserved": True, "constants": constants,
            "parameter_count": len(parameters), "parameters": [parameters[index] for index in sorted(parameters)],
            "parameter_vector": [parameters[index]["value"] for index in sorted(parameters)],
            "source_members": dict(sorted(members.items())), "source_tables": tables,
            "unused_source_members": sorted(members.keys() - referenced),
            "unmapped_source_tables": sorted(key for key, row in tables.items() if not row["mapped_parameter_indices"]),
            "source_inconsistencies": inconsistencies,
            "dispersion_metadata_note": "Column1 retained as sd; source headers also use variance and map_sigma_ptr says standard error. No statistical reinterpretation or confidence interval is inferred."}
