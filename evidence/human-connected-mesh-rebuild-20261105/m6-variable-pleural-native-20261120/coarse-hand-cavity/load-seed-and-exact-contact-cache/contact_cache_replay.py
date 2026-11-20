"""A/B the existing exact contact cache on the current complete substrate."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "operator-resident-030"))
import replay

replay.HERE = Path(sys.argv.pop(1)).resolve()

replay.MODES["exact-contact-cache"] = {
    **replay.PARALLEL_OPERATORS,
    "NUMI_HUMAN_CONTACT_ROW_CACHE": "1",
    "NUMI_HUMAN_CONTACT_AABB_EXCLUSION": "1",
    "NUMI_HUMAN_CONTACT_EXACT_MOTION_CACHE": "1",
    "NUMI_HUMAN_CONTACT_COMPACT_QUERIES": "1",
}
replay.main()
