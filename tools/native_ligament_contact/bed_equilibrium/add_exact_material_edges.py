"""Build a source variant of Matter's admission header; keep fallback bounds for every unproved edge."""
from pathlib import Path
import sys

def write(src, dest):
    text=src.read_text()
    pivot='[[nodiscard]] inline FEMReferenceDeterminantInterval femReferenceDisplacementDeterminantInterval('
    assert text.count(pivot)==1
    helper='''// Prove that the ordered TwoSum for a-b has a normal, one-word result.
// With a binade separation <= 25 the aligned FP32 significands occupy at
// most 49 bits (50 with addition), so the binary64 subtraction is exact.
// Require normal intermediate operands, including on FTZ-capable devices.
inline bool femExactWordDifference(float a, float b, float& value) noexcept {
    if (!std::isfinite(a) || !std::isfinite(b)) return false;
    if ((a != 0.0f && !std::isnormal(a)) || (b != 0.0f && !std::isnormal(b))) return false;
    if (a == b) { value = 0.0f; return true; }
    if (a == 0.0f || b == 0.0f) { value = a - b; return true; }
    if (std::abs(std::ilogb(a) - std::ilogb(b)) > 25) return false;
    const double exact = double(a) - double(b);
    if (std::abs(exact) > double(std::numeric_limits<float>::max())) return false;
    value = static_cast<float>(exact);
    return std::isnormal(value) && double(value) == exact;
}

inline bool femExactDisplacementEdge(float ah, float al, float bh, float bl, float& value) noexcept {
    float high, low;
    // Each component subtraction and their sum is exact. All following error
    // words of mrCompensatedAdd are zero; no absolute translation error enters.
    return femExactWordDifference(ah,bh,high) && femExactWordDifference(al,bl,low)
        && femExactWordDifference(high,-low,value);
}

'''
    text=text.replace(pivot,helper+pivot)
    pos='        safe=safe&&std::isfinite(du[3*r+c])&&std::isfinite(du32[3*r+c])&&'
    assert text.count(pos)==1
    text=text.replace(pos,'''        float exactEdge;
        if (femExactDisplacementEdge(displacementHigh[c+1][r], displacementLow[c+1][r],
                                     displacementHigh[0][r], displacementLow[0][r], exactEdge)) {
            du[3*r+c] = double(exactEdge);
            duError[3*r+c] = 0.0;
        }
'''+pos)
    dest.write_text(text)
if __name__=='__main__':write(Path(sys.argv[1]),Path(sys.argv[2]))
