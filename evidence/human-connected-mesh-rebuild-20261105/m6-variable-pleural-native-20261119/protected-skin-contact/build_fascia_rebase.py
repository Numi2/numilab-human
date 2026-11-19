"""Keep the weighted skin's contact witnesses in the mass-rebased body frames."""
import difflib
import os
import subprocess

from run_support_dependency_preflight import HERE, read, write, digest

base=HERE/"shared-fascia-pressure-build"
build=HERE/"shared-fascia-rebase-build"
build.mkdir()
unit="numilab_human_myosim_visual_probe.mm"
app="numi-human-native"
for p in (*base.glob("*.hpp"),base/"runtime.mm.o"):
    os.link(p,build/p.name)
(build/"metalrobo").mkdir()
for p in (base/"metalrobo").iterdir():
    os.link(p,build/"metalrobo"/p.name)
original=(base/unit).read_text()
updated=original
def replace(a,b):
    global updated
    assert updated.count(a)==1,a
    updated=updated.replace(a,b)
replace("""    LoadedSupportContacts& support, LoadedBones* bones, LoadedSkin* skin,
    std::string skinSourceMember,""",
"""    LoadedSupportContacts& support, LoadedBones* bones, LoadedSkin* weightedSkin,
    bool includeSkinFEM, std::string skinSourceMember,""")
replace("""            "resting passive fascia source/model is invalid");
    const bool includeOpticNerves""",
"""            "resting passive fascia source/model is invalid");
    auto* const skin=includeSkinFEM ? weightedSkin : nullptr;
    const bool includeOpticNerves""")
replace("""    rebaseNumiHumanSurfaceFrames(bones,skin,&tissues,anatomy,effectiveMassPartitions);""",
"""    // Detailed skin is also the contact owner for the weighted bed. Rebase it
    // along with the other surface frames even when no volumetric skin is cooked.
    rebaseNumiHumanSurfaceFrames(bones,weightedSkin,&tissues,anatomy,effectiveMassPartitions);""")
replace("""                        restingSkinFatFEMEnabled&&skinPayload?&*skinPayload:nullptr,
                        restingSkinFatFEMEnabled&&skinPayloadPath?""",
"""                        skinPayload?&*skinPayload:nullptr,restingSkinFatFEMEnabled,
                        restingSkinFatFEMEnabled&&skinPayloadPath?""")
(build/unit).write_text(updated)
(HERE/"shared-fascia-weighted-skin-rebase.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True),updated.splitlines(keepends=True),
    "a/apps/"+unit,"b/apps/"+unit)))
paths={str(base/f):str(build/f) for f in (app,app+".o",unit,"runtime.mm.o")}
commands=[[paths.get(a,a) for a in cmd] for cmd in read(base/"build-commands.json")[-2:]]
write(build/"build-commands.json",commands)
for cmd in commands:
    subprocess.run(cmd,cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",check=True)
print(digest(build/app),flush=True)
