"""One-shot, exact-source migration; removed after tests and publication."""
from pathlib import Path
import subprocess


def load(path: str, expected_blob: str) -> tuple[Path, str]:
    p = Path(path)
    actual = subprocess.check_output(['git', 'hash-object', str(p)], text=True).strip()
    if actual != expected_blob:
        raise RuntimeError('Source changed; reconcile before applying: ' + path)
    return p, p.read_text()


def once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError('Missing or ambiguous source anchor: ' + old[:100])
    return text.replace(old, new)


p, text = load('.numi/commands/human', 'becd0ab04034664dfbc0bf94f9ea48a69704f643')
start = text.index('# Canonical Numi Human v1 standing transaction')
end = text.index('# Canonical Numi-owned Human probe', start)
stand = text[start:end]
stand = once(stand,
    '# 51 exact source joint equalities, then executes assisted and zero-root-wrench\n# horizons in persistent Apple Metal.\n# Four native camera frames and a bitwise replay are mandatory evidence.',
    '# 51 exact source joint equalities, then executes without root assistance\n# using current-state passive joints. Assistance is an explicit diagnostic.\n# Rendering is optional; mechanics-only runs cannot qualify visual coverage.')
help_tail = '[--mechanics-overlay]'
assert stand.count(help_tail) == 2
stand = stand.replace(help_tail, '[--mechanics-overlay] [--mechanics-only] [--assisted-diagnostic]')
stand = once(stand, '    mechanics_overlay=false\n',
    '    mechanics_overlay=false\n    mechanics_only=false\n    assisted_diagnostic=false\n')
stand = once(stand, '            --mechanics-overlay)\n', '''            --mechanics-only)
                [ "$mechanics_only" = false ] || { printf '%s\\n' "--mechanics-only may be given only once" >&2; exit 2; }
                mechanics_only=true
                shift
                ;;
            --assisted-diagnostic)
                [ "$assisted_diagnostic" = false ] || { printf '%s\\n' "--assisted-diagnostic may be given only once" >&2; exit 2; }
                assisted_diagnostic=true
                shift
                ;;
            --mechanics-overlay)
''')
stand = once(stand, '    if [ -z "$soft_tissue_payload" ] && \\\n', '''    if [ "$mechanics_only" = true ] && [ "$mechanics_overlay" = true ]; then
        printf '%s\\n' "--mechanics-only cannot request a visual mechanics overlay" >&2
        exit 2
    fi
    if [ "$mechanics_only" = false ] && [ -z "$soft_tissue_payload" ] && \\
''')
stand = once(stand, '''        "$bone_payload" "$output_directory" \\
        --tendon-payload "$tendon_payload" \\
''', '''        "$output_directory"
    if [ "$mechanics_only" = false ]; then
        set -- "$probe" \\
            "$artifact_directory/myosim-fullbody-core-reference.nhrigid" \\
            "$artifact_directory/myosim-fullbody-muscle-reference.nhmyo" \\
            "$bone_payload" "$output_directory"
    fi
    set -- "$@" \\
        --tendon-payload "$tendon_payload" \\
''')
stand = once(stand, '''        --persistent-metal-stand \\
        --stand-root-assistance \\
        --stand-remove-assistance \\
        --stand-deterministic-replay \\
''', '''        --persistent-metal-stand \\
        --persistent-source-passive-joint-tissue \\
        --stand-contact-iterations 64 \\
        --persistent-stand-trace \\
        --stand-deterministic-replay \\
''')
stand = once(stand, '    if [ "$mechanics_overlay" = true ]; then\n', '''    if [ "$assisted_diagnostic" = true ]; then
        set -- "$@" --stand-root-assistance --stand-remove-assistance
    fi
    if [ "$mechanics_only" = true ]; then
        set -- "$@" --mechanics-only
    fi
    if [ "$mechanics_overlay" = true ]; then
''')
p.write_text(text[:start] + stand + text[end:])
subprocess.run(['sh', '-n', str(p)], check=True)

p, text = load('tests/test_importer.py', 'd146271ff2608670097b345520a819438303cfd0')
text = once(text, '                self.assertIn("--stand-remove-assistance", argv)\n',
    '                self.assertNotIn("--stand-root-assistance", argv)\n'
    '                self.assertNotIn("--stand-remove-assistance", argv)\n'
    '                self.assertIn("--persistent-source-passive-joint-tissue", argv)\n'
    '                self.assertIn("--persistent-stand-trace", argv)\n')
p.write_text(text)

p, text = load('src/numilab_human/native_force_convergence.py', '78ab83ce27e5a2bcaf3a852ad3e5f8e2e2f07130')
start = text.index('def _native_metrics(')
end = text.index('\n\ndef _field(', start)
replacement = '''def _native_metrics(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ImportError(f"cannot read native stdout {path}: {error}") from error
    kinds = (
        "myosim_articulated_marker_visual",
        "myosim_articulated_bodyparts_bone_visual",
        "myosim_articulated_mechanics",
    )
    matches = [(kind, line) for line in lines for kind in kinds
               if line.startswith(kind + "=")]
    if len(matches) != 1:
        raise ImportError(f"{path} must contain exactly one native result line")
    kind, line = matches[0]
    try:
        tokens = shlex.split(line)
    except ValueError as error:
        raise ImportError(f"{path} contains a malformed native result") from error
    values: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in values:
            raise ImportError(f"native result contains duplicate metric {key}")
        values[key] = value
    if values.get(kind) != "ok" or sum(key in values for key in kinds) != 1:
        raise ImportError(f"{path} does not contain an unambiguous successful native result")
    if kind == "myosim_articulated_mechanics":
        if (values.get("rendering_performed") != "false" or
                values.get("visual_coverage_qualified") != "false"):
            raise ImportError("mechanics-only result must explicitly exclude visual qualification")
    return values
'''
p.write_text(text[:start] + replacement + text[end:])
print('Applied scoped stand defaults and mechanics-summary compatibility')
