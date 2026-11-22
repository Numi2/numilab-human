"""Run the registered infeasible/feasible carrier fixture against the actual C++ flow code."""

from pathlib import Path
import json, subprocess, tempfile, sys


def check(owner, source, trace, fit):
    supply = next(
        json.loads(l.partition("=")[2])["supply"]
        for l in trace.read_text().splitlines()
        if l.startswith("cavity_carrier_route=")
    )
    mass = json.loads(fit.read_text())["proposal"]["mass_kg"]
    # Compile the exact owner functions without creating a second numerical implementation.
    raw = (owner / "NumiHumanSkinFatFixedCavityReconditioning.hpp").read_text()
    begin = raw.index("inline bool routeRestrictedParentMass(")
    fn = raw[begin : raw.index("\n}", begin) + 2]
    rows = ",".join(
        "{" + str(r["virtual_row"]) + "," + float(r["mass_kg"]).hex() + "}"
        for r in supply
    )
    values = ",".join(float(m).hex() for m in mass)
    ds = ",".join(
        "0.0" if i in r["allowed"] else "2.0" for r in supply for i in range(len(mass))
    )
    code = (
        '#include "NumiHumanConservativeSpatialMassTransport.hpp"\nnamespace metalrobo::human::coarse_tissue {\n'
        + fn
        + '\n#include "NumiHumanCavitySmallFlow.hpp"\n}\n'
    )
    code += (
        "int main(){using namespace metalrobo::human::coarse_tissue;std::map<uint32_t,double> src={"
        + rows
        + "};std::vector<double> m={"
        + values
        + "},d={"
        + ds
        + "};\n"
    )
    code += """
std::vector<std::map<uint32_t,double>> out;std::string error;
if(routeRestrictedParentMass(src,m,d,1.,out,error)){std::cerr<<"expected captured baseline failure";return 1;}
if(!routeCavityMassSmallFirst(src,m,d,1.,out,error)){std::cerr<<error;return 2;}
size_t r=0;for(auto [v,m0]:src){double s=0;for(size_t j=0;j<m.size();++j)if(out[j].contains(v)){
 if(d[r*m.size()+j]>1.||out[j][v]<=0.)return 3;s+=out[j][v];
}if(std::abs(s-m0)>1e-12*m0)return 4;++r;}
for(size_t j=0;j<m.size();++j){double s=0;for(auto [v,w]:out[j])s+=w;if(std::abs(s-m[j])>1e-12*m[j])return 5;}
std::fill_n(d.begin(),m.size(),2.);if(routeCavityMassSmallFirst(src,m,d,1.,out,error))return 6;
std::cout<<"carrier mass fixture passed "<<src.size()<<" sources "<<m.size()<<" sinks; disconnected source rejected\\n";
}
"""
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "test.cpp"
        binary = Path(tmp) / "test"
        f.write_text(code)
        subprocess.run(
            [
                "xcrun",
                "clang++",
                "-std=c++23",
                "-O2",
                "-I",
                str(owner),
                "-I",
                str(source),
                str(f),
                "-o",
                str(binary),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    check(*(Path(x).resolve() for x in sys.argv[1:]))
