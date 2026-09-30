#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct Vec { double x,y,z; };
static Vec sub(Vec a,Vec b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static double dot(Vec a,Vec b){return a.x*b.x+a.y*b.y+a.z*b.z;}

template <typename T> std::vector<T> read(const std::string& p) {
    std::ifstream in(p,std::ios::binary|std::ios::ate);
    if(!in) {std::fprintf(stderr,"cannot read %s\n",p.c_str());std::exit(2);}
    auto size=in.tellg();
    if(size<0 || size%sizeof(T)!=0) std::exit(3);
    std::vector<T> out(size/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(out.data()),size);
    if(!in) std::exit(4);
    return out;
}

static double sq(Vec d,Vec f,bool fec,double cv_f,double cv_t,double cv_fec) {
    double l2=dot(d,d);
    if(fec) return l2/(cv_fec*cv_fec);
    double proj=dot(d,f);
    return proj*proj/(cv_f*cv_f)+std::max(0.0,l2-proj*proj)/(cv_t*cv_t);
}

static double candidate(double tv[4],double dist[4][4],int target) {
    int p[3],k=0;
    for(int i=0;i<4;++i) if(i!=target) p[k++]=i;
    double best=std::numeric_limits<double>::infinity();
    for(int j=0;j<3;++j) best=std::min(best,tv[p[j]]+std::sqrt(dist[target][p[j]]));
    for(int i=0;i<3;++i) for(int j=i+1;j<3;++j) {
        int a=p[i],b=p[j];
        double A=dist[a][b], B=(A+dist[target][a]-dist[target][b])/2;
        double C=dist[target][a],dt=tv[b]-tv[a];
        double denominator=1-dt*dt/A;
        double numerator=C-B*B/A;
        if(denominator<=1e-12 || numerator<0) continue;
        double L=std::sqrt(std::max(0.0,numerator)/denominator);
        double u=(B-L*dt)/A;
        if(u>=0 && u<=1) best=std::min(best,tv[a]+dt*u+L);
    }
    int a=p[0],b=p[1],c=p[2];
    double A11=dist[a][b],A22=dist[a][c];
    double A12=(dist[a][b]+dist[a][c]-dist[b][c])/2;
    double B1=(A11+dist[target][a]-dist[target][b])/2;
    double B2=(A22+dist[target][a]-dist[target][c])/2;
    double det=A11*A22-A12*A12;
    if(det<=1e-25) return best;
    double inv11=A22/det,inv12=-A12/det,inv22=A11/det;
    double C=dist[target][a],dt1=tv[b]-tv[a],dt2=tv[c]-tv[a];
    double IB1=inv11*B1+inv12*B2,IB2=inv12*B1+inv22*B2;
    double IT1=inv11*dt1+inv12*dt2,IT2=inv12*dt1+inv22*dt2;
    double denominator=1-dt1*IT1-dt2*IT2;
    double numerator=C-B1*IB1-B2*IB2;
    if(denominator>1e-12 && numerator>=-1e-14) {
        double L=std::sqrt(std::max(0.0,numerator)/denominator);
        double u=IB1-L*IT1,v=IB2-L*IT2;
        if(u>=0 && v>=0 && u+v<=1)
            best=std::min(best,tv[a]+dt1*u+dt2*v+L);
    }
    return best;
}

int main(int argc,char**argv) {
    if(argc!=8) {
        std::fprintf(stderr,"usage: refine <asset> <scratch> <cv_f> <cv_t> <cv_fec> <stimulus_max_z> <fec_max_z>\n");
        return 2;
    }
    std::string base=argv[1],scratch=argv[2];
    double cv_f=std::strtod(argv[3],nullptr),cv_t=std::strtod(argv[4],nullptr),
           cv_fec=std::strtod(argv[5],nullptr),stimulus_z=std::strtod(argv[6],nullptr),
           fec_z=std::strtod(argv[7],nullptr);
    if(!(cv_f>0 && cv_t>0 && cv_fec>0 && stimulus_z>0 && fec_z>stimulus_z && fec_z<=1)) return 2;
    auto positions=read<Vec>(base+"/nodes.f64le");
    auto tets=read<uint32_t>(base+"/tetrahedra.u32le");
    auto labels=read<uint32_t>(base+"/labels.u32le");
    auto fibres=read<Vec>(base+"/fibres.f64le");
    auto rho=read<double>(base+"/uvc_rho.f64le");
    auto z=read<double>(base+"/uvc_z.f64le");
    auto source_nodes=read<uint32_t>(scratch+"/ventricular-source-nodes.u32le");
    auto dof_regions=read<uint32_t>(scratch+"/ventricular-dof-regions.u32le");
    auto arrival=read<double>(scratch+"/graph-arrival.f64le");
    if(tets.size()!=labels.size()*4 || fibres.size()!=labels.size() ||
       source_nodes.size()!=arrival.size() || dof_regions.size()!=arrival.size() ||
       rho.size()!=positions.size() || z.size()!=positions.size()) return 3;
    std::vector<int32_t> lut(positions.size(),-1);
    for(size_t i=0;i<source_nodes.size();++i) {
        if(source_nodes[i]>=positions.size() || !std::isfinite(arrival[i]) || arrival[i]<0 ||
           (dof_regions[i]!=0 && dof_regions[i]!=2)) return 4;
        if(dof_regions[i]==0) {
            if(lut[source_nodes[i]]!=-1) return 4;
            lut[source_nodes[i]]=i;
        }
    }
    std::vector<int32_t> rv_lut=lut;
    for(size_t i=0;i<source_nodes.size();++i) if(dof_regions[i]==2) {
        if(lut[source_nodes[i]]<0 || rv_lut[source_nodes[i]]!=lut[source_nodes[i]]) return 4;
        rv_lut[source_nodes[i]]=i;
    }
    struct Cell {std::array<uint32_t,4> n;std::array<double,6> edge;};
    std::vector<Cell> cells;cells.reserve(1100000);
    constexpr int pairs[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for(size_t i=0;i<labels.size();++i) if(labels[i]==1 || labels[i]==2) {
        Cell cell;
        Vec f=fibres[i], p[4];
        double fn=std::sqrt(dot(f,f));f={f.x/fn,f.y/fn,f.z/fn};
        bool endo=false;double maxz=-1e100;
        for(int j=0;j<4;++j){
            uint32_t source=tets[4*i+j];
            if(source>=positions.size()) return 4;
            int32_t owner=(labels[i]==1?lut[source]:rv_lut[source]);
            if(owner<0) return 4;
            cell.n[j]=owner;p[j]=positions[source];
            endo|=rho[source]==0;maxz=std::max(maxz,z[source]);
        }
        bool fec=endo && maxz<=fec_z;
        for(int j=0;j<6;++j)
            cell.edge[j]=sq(sub(p[pairs[j][0]],p[pairs[j][1]]),f,fec,cv_f,cv_t,cv_fec);
        cells.push_back(cell);
    }
    std::vector<uint32_t> order(cells.size());std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){
        auto min_time=[&](uint32_t i){
            double m=arrival[cells[i].n[0]];
            for(int j=1;j<4;++j)m=std::min(m,arrival[cells[i].n[j]]);
            return m;
        };
        double x=min_time(a),y=min_time(b);
        return x==y ? a<b : x<y;
    });
    for(size_t i=0;i<source_nodes.size();++i)
        if(rho[source_nodes[i]]==0 && z[source_nodes[i]]>=0 && z[source_nodes[i]]<=stimulus_z) arrival[i]=0;
    std::printf("cells %zu nodes %zu\n",cells.size(),arrival.size());std::fflush(stdout);
    bool converged=false;
    for(int sweep=0;sweep<40;++sweep) {
        double max_drop=0,sum_drop=0;uint64_t changes=0;
        for(int direction=0;direction<2;++direction) for(size_t at=0;at<order.size();++at) {
            const Cell& cell=cells[order[direction==0?at:order.size()-1-at]];
            double distances[4][4]{};
            for(int j=0;j<6;++j) distances[pairs[j][0]][pairs[j][1]]=distances[pairs[j][1]][pairs[j][0]]=cell.edge[j];
            double times[4];for(int j=0;j<4;++j) times[j]=arrival[cell.n[j]];
            for(int target=0;target<4;++target){
                uint32_t src=source_nodes[cell.n[target]];
                if(rho[src]==0 && z[src]>=0 && z[src]<=stimulus_z) continue;
                double other=std::numeric_limits<double>::infinity();
                for(int j=0;j<4;++j) if(j!=target) other=std::min(other,times[j]);
                if(other>=times[target]) continue;
                double next=candidate(times,distances,target);
                if(next<times[target]-1e-14){
                    double drop=times[target]-next;
                    max_drop=std::max(max_drop,drop);sum_drop+=drop;++changes;
                    times[target]=next;arrival[cell.n[target]]=next;
                }
            }
        }
        std::printf("sweep %d changes %llu max_drop_ms %.9g sum_drop_ms %.9g\n",sweep+1,
                    (unsigned long long)changes,max_drop*1000,sum_drop*1000);std::fflush(stdout);
        if(max_drop<1e-10) {converged=true;break;}
    }
    if(!converged) {std::fprintf(stderr,"tetrahedral sweep did not converge\n");return 6;}
    std::ofstream out(scratch+"/refined-arrival.f64le",std::ios::binary);
    out.write(reinterpret_cast<const char*>(arrival.data()),arrival.size()*sizeof(double));
    return out?0:5;
}
