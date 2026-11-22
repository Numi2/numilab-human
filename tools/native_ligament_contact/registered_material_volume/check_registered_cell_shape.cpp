#include "NumiHumanMaterialCellShapeAudit.hpp"
#include <cassert>

int main(){
    using namespace numiHumanCellShape;
    const Vec b{1.0,0.0,0.0},c{0.0,1.0,0.0},d{0.0,0.0,1.0},zero{};
    const double expected=1.0/std::sqrt(8.0);
    assert(std::abs(normalizedDeterminant(1.0,b,c,d)-expected)<1e-15);
    assert(normalizedDeterminant(-1.0,c,b,d)<0.0);
    assert(normalizedDeterminant(0.0,zero,zero,zero)==0.0);
    Summary result;
    result.observe(1.0,expected,0);
    result.observe(-1.0,-expected,1);
    result.observe(6.0e-18,1.0e-5,2);
    result.observe(std::numeric_limits<double>::quiet_NaN(),0,3);
    assert(result.cells==4 && result.inverted==1 && result.belowVolume==2 &&
           result.belowShape==1 && result.nonfinite==1 && result.minimumShapeRow==1);
}
