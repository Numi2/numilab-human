"""Verify actual M6 admission, exact rational memberships and terminal receipt parity."""

import json, subprocess, sys
from pathlib import Path
import numpy as np
from test_heart_gate import inside
from certify_heart import sha
from build_drive_heart import tables


def run(before, case, source, compiled):
    sets = [tables(p) for p in (before, case)]
    rows = [
        np.fromfile(p / "coordinates.bin", "<f4").reshape(-1, 16)
        for p in (before, case)
    ]
    assert len(rows[0]) == len(rows[1]) == 30000
    rng = np.random.default_rng(8358)
    tests = []
    for coords, parents in sets:
        for d in coords["domains"]:
            a, b, lo, hi = np.asarray(d["tube"])
            for t in (-0.03, 0.0, 0.3, 0.7, 1.0, 1.03):
                for j in range(12):
                    x = rng.uniform(lo, hi) if j % 2 else rng.uniform(2 * lo, 2 * hi)
                    tests.append(a + t * (b - a) + x)
    tests.extend(np.eye(7) * 2)
    bad = np.zeros((len(tests) + 3, 16), "<f4")
    bad[:-3, :7] = tests
    bad[-3:, 0] = [np.nan, -np.inf, np.inf]
    bad.view("<u4")[:, 8] = 5
    inputs = np.concatenate([*rows, bad])
    ip = case / "test-coordinates.bin"
    op = case / "test-domains.bin"
    inputs.tofile(ip)
    subprocess.run(
        [
            source / "cardiac-tube-membership-probe",
            compiled / "test-membership.metallib",
            ip,
            before / "unused-tube.bin",
            op,
        ],
        check=True,
    )
    code = np.fromfile(op, "<u4")
    reject = 0xFFFFFFFF
    assert (
        len(code) == len(inputs)
        and (code[:60000] != reject).all()
        and (code[-10:] == reject).all()
    )
    prev = np.fromfile(case / "prior-gate.bin", "<u4")
    assert (
        np.array_equal(prev == reject, (code[30000:60000] & 0xF0000000) == 0x50000000)
        and (code[:30000] < 0x50000000).all()
    )
    verified = 0
    for point, value in zip(inputs, code, strict=True):
        c = int(value)
        if c in (reject, 0, 1):
            continue
        tag, index = c >> 28, c & 0xFFFFFFF
        if tag not in (4, 5):
            raise ValueError("Invalid accepted ID")
        domains, proof = sets[tag - 4]
        parents = domains["domains"]
        pr = proof["parents"]
        if index < len(pr):
            assert pr[index]["parent_accepted_without_children"] and inside(
                point[:7], parents[index]
            )
        else:
            child = index - len(pr)
            for parent in pr:
                if child < len(parent["children"]):
                    assert inside(point[:7], parents[parent["parent"]]) and inside(
                        point[:7], parent["children"][child]
                    )
                    break
                child -= len(parent["children"])
            else:
                raise ValueError("Unknown child")
        verified += 1
    # Exercise the actual export helper, including malformed witnesses.
    test = case / "terminal-domain-test.cpp"
    test.write_text(
        """#include <cmath>
#include <array>
#include <iterator>
#include <fstream>
#include <cstdint>
#include "NumiHumanTerminalCardiacDomains.hpp"
int main(int argc,char** argv) {
    if(argc!=3) return 1;
    std::ifstream a(argv[1],std::ios::binary),b(argv[2],std::ios::binary);
    std::array<float,16> rec;
    while(a.read(reinterpret_cast<char*>(rec.data()),64)) {
        uint32_t id;
        if(!b.read(reinterpret_cast<char*>(&id),4)) return 2;
        std::array<float,7> q;std::copy_n(rec.data(),7,q.data());
        if(id>=0x40000000u && id!=0xffffffffu && !numiHumanTerminalNestedDomain(id,q)) return 3;
        if(numiHumanTerminalNestedDomain(0x5fffffffu,q)) return 4;
        q[0]=NAN;
        if(numiHumanTerminalNestedDomain(id,q)) return 5;
    }
    return !a.eof() || b.peek()!=std::char_traits<char>::eof();
}""".replace("#include <array>", "#include <array>\n#include <algorithm>")
    )
    host = case / "test-terminal-domains"
    subprocess.run(
        ["/usr/bin/clang++", "-std=c++20", "-O2", "-I", compiled, test, "-o", host],
        check=True,
    )
    subprocess.run([host, ip, op], check=True)
    results = dict(
        baseline_states=30000,
        treatment_forecast_states=30000,
        additional_treatment_admissions=int((prev == reject).sum()),
        rational_nested_admissions=verified,
        random_and_adversarial_inputs=len(bad),
        rejected_adversarial=int((code[60000:] == reject).sum()),
        outside_and_nonfinite_reject=True,
        terminal_receipt_parity=True,
        device_gate=sha(compiled / "test-membership.metallib"),
        source=sha(compiled / "drive_cardiac_subdomains.metalinc"),
    )
    (case / "gpu-membership.json").write_text(json.dumps(results, indent=2) + "\n")
    print(results)


if __name__ == "__main__":
    run(*(Path(p).resolve() for p in sys.argv[1:]))
