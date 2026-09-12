#include "numi/matter/matter.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

// Independent C++ admission of the offline source frame field into the native
// authoring value type. This does not cook/step a world or supply tissue density.
namespace {
using Vec = std::array<double, 3>;
using Matrix = std::array<double, 9>;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
Vec unit(Vec v) {
    const double length = std::hypot(v[0], v[1], v[2]);
    require(std::isfinite(length) && length > 0, "invalid source axis");
    for (auto& x : v) { require(std::isfinite(x), "nonfinite source axis"); x /= length; }
    return v;
}
double dot(const Vec& a, const Vec& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vec cross(const Vec& a, const Vec& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}
template<class T> std::array<T, 9> rotation(const std::array<T, 4>& q) {
    const auto [x,y,z,w] = q;
    return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)};
}
template<class T> T row(std::ifstream& stream) {
    T value{}; stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    require(bool(stream), "truncated frame field"); return value;
}
unsigned digit(char c) {
    if (c >= '0' && c <= '9') return unsigned(c-'0');
    if (c >= 'a' && c <= 'f') return unsigned(c-'a'+10);
    throw std::runtime_error("source identity must be lowercase SHA256");
}
}
int main(int argc, char** argv) {
    try {
        require(argc==6, "usage: cardiac-material-frame-check FIBRES SHEETS QUATERNIONS CELLS SOURCE_SHA256");
        require(std::endian::native==std::endian::little, "little-endian host required");
        const std::string countArgument(argv[4]); std::size_t parsed=0;
        const auto count = std::stoull(countArgument,&parsed);
        require(parsed==countArgument.size() && count>0 && count<=std::numeric_limits<std::uint32_t>::max(), "invalid cell count");
        require(std::filesystem::file_size(argv[1])==count*sizeof(Vec)
            && std::filesystem::file_size(argv[2])==count*sizeof(Vec)
            && std::filesystem::file_size(argv[3])==count*sizeof(std::array<double,4>), "field shape mismatch");
        std::ifstream fibres(argv[1],std::ios::binary), sheets(argv[2],std::ios::binary), quaternions(argv[3],std::ios::binary);
        require(bool(fibres)&&bool(sheets)&&bool(quaternions), "field open failed");
        numi::matter::ObjectSource native;
        native.representation=numi::matter::Representation::fem;
        native.femMaterialFrameRotations.reserve(count);
        const std::string identity(argv[5]); require(identity.size()==64, "source identity length");
        for (std::size_t byte=0; byte<32; ++byte)
            native.femMaterialFrameSourceIdentity[byte/8] |=
                std::uint64_t(16*digit(identity[2*byte])+digit(identity[2*byte+1])) << (8*(byte%8));
        require(std::any_of(native.femMaterialFrameSourceIdentity.begin(),native.femMaterialFrameSourceIdentity.end(),
            [](auto word) { return word!=0; }), "empty source identity");
        double maximumNorm=0, maximumBasis=0, maximumCookedNorm=0, maximumCookedBasis=0;
        for (std::size_t cell=0; cell<count; ++cell) {
            const Vec f=unit(row<Vec>(fibres)); const Vec initialSheet=unit(row<Vec>(sheets));
            const double projection=dot(f,initialSheet);
            Vec s{}; for (std::size_t k=0; k<3; ++k) s[k]=initialSheet[k]-projection*f[k];
            require(std::hypot(s[0],s[1],s[2])>1e-8, "degenerate sheet projection");
            s=unit(s); const Vec n=unit(cross(f,s));
            const Matrix expected{f[0],s[0],n[0],f[1],s[1],n[1],f[2],s[2],n[2]};
            const auto q=row<std::array<double,4>>(quaternions);
            double norm=0;
            for (const auto value:q) {
                require(std::isfinite(value), "nonfinite quaternion");
                require(value!=0 || !std::signbit(value), "noncanonical negative zero");
                norm+=value*value;
            }
            const std::array<double,4> signOrder{q[3],q[0],q[1],q[2]};
            const auto first=std::find_if(signOrder.begin(),signOrder.end(),[](double v){return v!=0;});
            require(first!=signOrder.end() && *first>0, "noncanonical quaternion sign");
            maximumNorm=std::max(maximumNorm,std::abs(norm-1));
            const auto matrix=rotation(q);
            for (std::size_t k=0; k<9; ++k) maximumBasis=std::max(maximumBasis,std::abs(matrix[k]-expected[k]));
            std::array<float,4> cooked{};
            double cookedNorm=0; float squared=0;
            for (std::size_t k=0; k<4; ++k) {
                cooked[k]=float(q[k]); cookedNorm+=double(cooked[k])*cooked[k]; squared+=cooked[k]*cooked[k];
            }
            maximumCookedNorm=std::max(maximumCookedNorm,std::abs(cookedNorm-1));
            const float inverse=1/std::sqrt(squared);
            for (auto& value:cooked) value*=inverse;
            const auto cookedMatrix=rotation(cooked);
            for (std::size_t k=0; k<9; ++k) maximumCookedBasis=std::max(maximumCookedBasis,std::abs(double(cookedMatrix[k])-expected[k]));
            native.femMaterialFrameRotations.push_back(q);
        }
        require(maximumNorm<=1e-12 && maximumBasis<=1e-12, "source frame reconstruction outside declared policy");
        require(maximumCookedNorm<=16*std::numeric_limits<float>::epsilon(), "frame rejected by native FP32 norm envelope");
        require(maximumCookedBasis<=1e-6, "cooked frame changed source basis beyond FP32 gate");
        std::cout << std::setprecision(17) << "{\"status\":\"pass\",\"cells\":" << native.femMaterialFrameRotations.size()
            << ",\"native_abi\":" << NM_MATTER_ABI_VERSION << ",\"maximum_quaternion_norm_error\":" << maximumNorm
            << ",\"maximum_source_basis_error\":" << maximumBasis << ",\"maximum_cooked_norm_error\":" << maximumCookedNorm
            << ",\"maximum_cooked_basis_error\":" << maximumCookedBasis << ",\"source_identity\":\"" << identity
            << "\",\"physical_steps\":0,\"world_cooked\":false,\"anatomical_mechanics_qualified\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardiac material frame check failed: " << error.what() << '\n'; return 1;
    }
}
