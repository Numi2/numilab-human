// Source-mesh active-fibre internal residual at the fixed CT geometry.
// Offline reference only: no Matter accepted step or cardiac motion occurs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

struct Vec { double x,y,z; };
static_assert(sizeof(Vec)==3*sizeof(double));
static Vec operator+(Vec a,Vec b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static Vec operator-(Vec a,Vec b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static Vec operator*(Vec a,double s){return {a.x*s,a.y*s,a.z*s};}
static Vec& operator+=(Vec& a,Vec b){a=a+b;return a;}
static double dot(Vec a,Vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static Vec cross(Vec a,Vec b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}

template <typename T> static std::vector<T> read(const std::string& path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file) {std::fprintf(stderr,"cannot open %s\n",path.c_str());std::exit(2);}
    auto size=file.tellg();
    if(size<0 || size%sizeof(T)) std::exit(2);
    std::vector<T> out(size/sizeof(T));
    file.seekg(0);file.read(reinterpret_cast<char*>(out.data()),size);
    if(!file) std::exit(2);
    return out;
}

static double tanh_stress(double time_ms,double activation_ms,double peak_pa,
                          double delay_ms,double contraction_ms,
                          double relaxation_ms,double duration_ms) {
    double elapsed=time_ms-activation_ms-delay_ms;
    if(!(elapsed>0 && elapsed<duration_ms)) return 0;
    double rise=std::tanh(elapsed/contraction_ms);
    double fall=std::tanh((duration_ms-elapsed)/relaxation_ms);
    return peak_pa*rise*rise*fall*fall;
}

int main(int argc,char**argv) {
    if(argc!=10) {
        std::fprintf(stderr,"usage: active_force <asset> <activation> <force_output> <frame_ms> <peak_pa> <delay_ms> <contraction_ms> <relaxation_ms> <duration_ms>\n");
        return 2;
    }
    const std::string asset=argv[1], activation=argv[2], output=argv[3];
    double time_ms=std::strtod(argv[4],nullptr),peak_pa=std::strtod(argv[5],nullptr),
           delay_ms=std::strtod(argv[6],nullptr),contraction_ms=std::strtod(argv[7],nullptr),
           relaxation_ms=std::strtod(argv[8],nullptr),duration_ms=std::strtod(argv[9],nullptr);
    if(!(std::isfinite(time_ms) && std::isfinite(peak_pa) &&
         std::isfinite(delay_ms) && std::isfinite(contraction_ms) &&
         std::isfinite(relaxation_ms) && std::isfinite(duration_ms) &&
         time_ms>=0 && peak_pa>0 && delay_ms>=0 && contraction_ms>0 &&
         relaxation_ms>0 && duration_ms>0)) return 2;
    auto positions=read<Vec>(asset+"/nodes.f64le");
    auto tets=read<uint32_t>(asset+"/tetrahedra.u32le");
    auto labels=read<uint32_t>(asset+"/labels.u32le");
    auto fibres=read<Vec>(asset+"/fibres.f64le");
    auto source_nodes=read<uint32_t>(activation+"/ventricular-source-nodes.u32le");
    auto regions=read<uint32_t>(activation+"/ventricular-dof-regions.u32le");
    auto arrival=read<double>(activation+"/refined-arrival.f64le");
    if(positions.size()!=300965 || labels.size()!=1470083 || tets.size()!=labels.size()*4 ||
       fibres.size()!=labels.size() || source_nodes.size()!=arrival.size() ||
       regions.size()!=arrival.size() || source_nodes.size()!=218080) return 3;
    const size_t mechanical_nodes=source_nodes.size()-3;
    for(size_t i=0;i<source_nodes.size();++i) {
        if((i<mechanical_nodes && regions[i]!=0) ||
           (i>=mechanical_nodes && regions[i]!=2) ||
           (i>0 && i<mechanical_nodes && source_nodes[i]<=source_nodes[i-1])) return 3;
    }
    std::vector<int32_t> source_lut(positions.size(),-1);
    for(size_t i=0;i<source_nodes.size();++i) {
        if(source_nodes[i]>=positions.size() || !std::isfinite(arrival[i]) || arrival[i]<0) return 3;
        if(regions[i]==0) {
            if(source_lut[source_nodes[i]]!=-1) return 3;
            source_lut[source_nodes[i]]=i;
        } else if(regions[i]!=2) return 3;
    }
    std::vector<int32_t> rv_lut=source_lut;
    for(size_t i=0;i<source_nodes.size();++i) if(regions[i]==2) {
        if(source_lut[source_nodes[i]]<0 || rv_lut[source_nodes[i]]!=source_lut[source_nodes[i]]) return 3;
        rv_lut[source_nodes[i]]=i;
    }
    // The first 218077 DOFs are exact source-node identities; only the final
    // three are duplicated RV electrical IDs. Mechanics keeps source IDs.
    // Positive internal residual R_i = integral P grad(N_i) dV. A native
    // external nodal load has the opposite sign and needs an accepted owner.
    std::vector<Vec> force(mechanical_nodes,{0,0,0});
    const double qa=(1+3/std::sqrt(5.0))/4;
    const double qb=(1-qa)/3;
    double integrated_tension=0, volume_sum=0, max_tension=0;
    uint64_t count=0, active=0;
    for(size_t i=0;i<labels.size();++i) if(labels[i]==1 || labels[i]==2) {
        uint32_t id[4];Vec p[4];double t[4];
        for(int j=0;j<4;++j) {
            id[j]=tets[i*4+j];
            if(id[j]>=positions.size() || source_lut[id[j]]<0) return 4;
            int dof=labels[i]==1?source_lut[id[j]]:rv_lut[id[j]];
            if(dof<0) return 4;
            p[j]=positions[id[j]];t[j]=arrival[dof]*1000;
        }
        Vec e1=p[1]-p[0],e2=p[2]-p[0],e3=p[3]-p[0];
        double det=dot(e1,cross(e2,e3));
        if(!(det>0 && std::isfinite(det))) return 4;
        double volume=det/6;
        Vec f=fibres[i];double norm=std::sqrt(dot(f,f));
        if(!(norm>0 && std::isfinite(norm))) return 4;
        f=f*(1/norm);
        double sum_t=t[0]+t[1]+t[2]+t[3],stress=0;
        for(int j=0;j<4;++j)
            stress+=tanh_stress(time_ms,qb*sum_t+(qa-qb)*t[j],peak_pa,
                                delay_ms,contraction_ms,relaxation_ms,duration_ms)/4;
        Vec grad[4];
        grad[1]=cross(e2,e3)*(1/det);
        grad[2]=cross(e3,e1)*(1/det);
        grad[3]=cross(e1,e2)*(1/det);
        grad[0]=(grad[1]+grad[2]+grad[3])*(-1);
        for(int j=0;j<4;++j)
            force[source_lut[id[j]]]+=f*(volume*stress*dot(f,grad[j]));
        ++count;if(stress>0) ++active;
        volume_sum+=volume;integrated_tension+=volume*stress;
        max_tension=std::max(max_tension,stress);
    }
    std::ofstream file(output,std::ios::binary);
    file.write(reinterpret_cast<const char*>(force.data()),force.size()*sizeof(Vec));
    if(!file) return 5;
    std::printf("cells=%llu active_cells=%llu nodes=%zu volume_m3=%.17g stress_volume_integral_j=%.17g max_tension_pa=%.17g\n",
                (unsigned long long)count,(unsigned long long)active,force.size(),
                volume_sum,integrated_tension,max_tension);
    return 0;
}
