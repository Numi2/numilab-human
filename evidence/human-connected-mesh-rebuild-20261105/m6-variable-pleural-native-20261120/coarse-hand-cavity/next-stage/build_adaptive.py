"""Build the certified adaptive Newton schedule on the exact latest Human candidate."""
from pathlib import Path
import sys

directory=Path(__file__).resolve().parent
sys.path.insert(0,str(directory.parent))
from build_hand_trial import APP, HERE, build

source=(HERE/"terminal-contact-native-build"/APP).read_text()
updated=source
edits=[
    ("        [[maybe_unused]] std::size_t lastStagedSolverTier = 0u;",
     "        [[maybe_unused]] unsigned stagedSolverHoldSteps = 0u;"),
    ("                constexpr std::uint32_t shortBudget = 2u;",
     "                constexpr std::uint32_t shortBudget = 4u;"),
    ("""                // A contact event that needed the intermediate tier often
                // persists into the next physical step. Restart each new
                // horizon with the shortest budget.
                const std::size_t startTier = lastStagedSolverTier == 0u ? 0u : 1u;""",
     """                // Keep two accepted intermediate starts after an escalation.
                // Recovery to the shortest budget avoids staying at eight
                // Newton passes for the lifetime of the settled state.
                const std::size_t startTier = stagedSolverHoldSteps == 0u ? 0u : 1u;"""),
    ("                        lastStagedSolverTier = tier;",
     """                        if (tier > startTier)
                            stagedSolverHoldSteps = 2u;
                        else if (stagedSolverHoldSteps != 0u)
                            --stagedSolverHoldSteps;"""),
]
for old,new in edits:
    assert updated.count(old)==1
    updated=updated.replace(old,new)
output=directory/"adaptive-build"
output.mkdir()
build(source,updated,output,output/"adaptive.patch")
