"""Extend the existing cardiac geometry owner with source-certified pressure domains."""

from pathlib import Path
import json, shutil, subprocess, sys
from build_qualified_heart import emit
from check_heart_proof import check
from certify_heart import sha


def replace(s, old, new):
    if s.count(old) != 1:
        raise ValueError("Unexpected source near " + old[:70])
    return s.replace(old, new)


def tables(proof):
    return [
        json.loads((proof / f).read_text())
        for f in ("four-domains.json", "refinement.json")
    ]


def terminal_table(domain, proofs):
    rows = domain["domains"]
    parents = proofs["parents"]
    children = [child for parent in parents for child in parent["children"]]
    owners = [i for i, r in enumerate(parents) for _ in r["children"]]

    def row(data):
        field = lambda v: "{" + ",".join(float(x).hex() + "f" for x in v) + "}"
        return (
            "{"
            + ",".join(field(v) for v in data["tube"])
            + ","
            + str(data["pivot"])
            + "}"
        )

    return (
        "static constexpr NumiTerminalCardiacTube rows[]={"
        + ",\n".join(row(r) for r in rows + children)
        + "};\n"
        + "static constexpr bool complete[]={"
        + ",".join(
            "true" if p["parent_accepted_without_children"] else "false"
            for p in parents
        )
        + "};\n"
        + "static constexpr unsigned parentOf[]={"
        + ",".join(map(str, owners or [0]))
        + "};\n"
        + "if (index >= std::size(rows)) return false;\n"
        + "if (index < std::size(complete)) return complete[index] && member(rows[index]);\n"
        + "return member(rows[parentOf[index-std::size(complete)]]) && member(rows[index]);\n"
    )


def host_receipt(baseline, extended):
    start = """struct NumiTerminalCardiacTube { float a[7], b[7], lo[7], hi[7]; int p; };
inline bool numiHumanTerminalNestedDomain(unsigned domain,const std::array<float,7>& q) {
    const auto member=[&](const NumiTerminalCardiacTube& v) {
        for(float x:q) if(!std::isfinite(x)) return false;
        const double t=(double(q[v.p])-v.a[v.p])/(double(v.b[v.p])-v.a[v.p]);
        if(!std::isfinite(t)||t<0.||t>1.) return false;
        for(int j=0;j<7;++j) {
            if(j==v.p) continue;
            const double difference=double(q[j])-v.a[j]-t*(double(v.b[j])-v.a[j]);
            if(!std::isfinite(difference)||difference<v.lo[j]||difference>v.hi[j]) return false;
        }
        return true;
    };
"""
    for tag, t in [("0x40000000u", baseline), ("0x50000000u", extended)]:
        start += (
            "if((domain&0xf0000000u)=="
            + tag
            + "){\nconst auto index=domain-"
            + tag
            + ";\n"
            + terminal_table(*t)
            + "}\n"
        )
    return start + "return false;\n}\n"


def build(original, previous, work, source, app, out):
    oldhash = json.loads((original / "identities.json").read_text())
    for file, h in oldhash.items():
        if sha(original / file) != h:
            raise ValueError("Original owner changed: " + file)
    proof0, proof1 = check(previous, source), check(work, source)
    t0, t1 = tables(previous), tables(work)
    old = emit(*t0)
    if old != (original / "native_cardiac_subdomains.metalinc").read_text():
        raise ValueError("Mismatched baseline domain table")
    out.mkdir(exist_ok=False)
    (out / "metalrobo").mkdir()
    for f in [
        *original.glob("*.hpp"),
        *original.glob("*.mm"),
        *original.glob("*.metalinc"),
        original / "metalrobo/MetalRobo.metallib",
    ]:
        shutil.copyfile(f, out / f.relative_to(original))
    more = emit(*t1)
    for key in (
        "heartParentDomains",
        "heartChildDomains",
        "heartChildSpans",
        "cardiacNestedDomain",
    ):
        more = more.replace(key, "drive" + key[0].upper() + key[1:])
    (out / "drive_cardiac_subdomains.metalinc").write_text(more)
    m = (original / "human_respiration.metal").read_text()
    m = replace(
        m,
        '#include "native_cardiac_subdomains.metalinc"',
        '#include "native_cardiac_subdomains.metalinc"\n#include "drive_cardiac_subdomains.metalinc"',
    )
    a = m.index("matched=0x40000000u+witness;")
    b = m.index("    output[0].first=", a)
    m = (
        m[:b]
        + """    if(status==5u) {
        uint witness=0;
        if(driveCardiacNestedDomain(coordinates,witness)) {
            matched=0x50000000u+witness;
            status=0u;
        }
    }
"""
        + m[b:]
    )
    (out / "human_respiration.metal").write_text(m)
    terminal = out / "NumiHumanRestingCoupling.hpp"
    term = terminal.read_text()
    (out / "NumiHumanTerminalCardiacDomains.hpp").write_text(host_receipt(t0, t1))
    term = replace(
        term,
        "inline bool numiHumanRestingTerminalDomain(",
        '#include "NumiHumanTerminalCardiacDomains.hpp"\ninline bool numiHumanRestingTerminalDomain(',
    )
    term = replace(
        term,
        "    if(domain<boxCount) return true;",
        "    if(domain>=0x40000000u) return numiHumanTerminalNestedDomain(domain,q);\n    if(domain<boxCount) return true;",
    )
    terminal.write_text(term)
    cmds = [
        [arg.replace(str(original) + "/", str(out) + "/") for arg in cmd]
        for cmd in json.loads((original / "build-commands.json").read_text())
    ]
    (out / "build-commands.json").write_text(json.dumps(cmds, indent=2) + "\n")
    for cmd in cmds:
        subprocess.run(cmd, cwd=app / "build-native", check=True)
    # Compile the same membership functions and order used by the integrated owner.
    test = (
        (original / "test-membership.metal")
        .read_text()
        .replace(
            '#include "native_cardiac_subdomains.metalinc"',
            '#include "native_cardiac_subdomains.metalinc"\n#include "drive_cardiac_subdomains.metalinc"',
        )
    )
    test = replace(
        test,
        "code=2u+id;",
        "code=0x40000000u+id;\n        if(code==MR_INVALID_INDEX && driveCardiacNestedDomain(q,id)) code=0x50000000u+id;",
    )
    (out / "test-membership.metal").write_text(test)
    subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            "-fno-fast-math",
            "-I",
            str(app / "include"),
            "-c",
            out / "test-membership.metal",
            "-o",
            out / "test-membership.air",
        ],
        check=True,
    )
    subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metallib",
            out / "test-membership.air",
            "-o",
            out / "test-membership.metallib",
        ],
        check=True,
    )
    names = list(oldhash) + [
        "drive_cardiac_subdomains.metalinc",
        "NumiHumanTerminalCardiacDomains.hpp",
        "NumiHumanRestingCoupling.hpp",
        "test-membership.metallib",
    ]
    (out / "identities.json").write_text(
        json.dumps({f: sha(out / f) for f in names}, indent=2) + "\n"
    )
    print(dict(baseline=proof0, updated=proof1), flush=True)


if __name__ == "__main__":
    build(*(Path(p).resolve() for p in sys.argv[1:]))
