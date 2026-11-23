// Reuse the registered cortical surface BVH for independent skin-enclosure rays.
#include "cortical_distance.cpp"

static bool onRay(const Box& box, V p, V d) {
    double first = 0, last = INFINITY;
    for (int axis=0; axis<3; ++axis) {
        double a=(box.lo[axis]-p[axis])/d[axis];
        double b=(box.hi[axis]-p[axis])/d[axis];
        if (a>b) std::swap(a,b);
        first=std::max(first,a);
        last=std::min(last,b);
    }
    return first<=last;
}

// Return -1 when the ray grazes a triangle edge or begins on a surface.
// Classifying that ray by an arbitrary tie break could change parity.
static int crossings(const Surface& s, int node, V p, V d) {
    const auto& n=s.tree[node];
    if (!onRay(n.box,p,d)) return 0;
    if (n.left>=0) {
        int a=crossings(s,n.left,p,d), b=crossings(s,n.right,p,d);
        return a<0 || b<0 ? -1 : a+b;
    }
    int hits=0;
    for (int i=n.first;i<n.end;++i) {
        auto t=s.t[s.indices[i]];
        auto a=s.x[t[0]],e=sub(s.x[t[1]],a),g=sub(s.x[t[2]],a);
        auto h=cross(d,g);double det=dot(e,h);
        if (std::abs(det)<1e-22) continue;
        auto r=sub(p,a);auto q=cross(r,e);
        double u=dot(r,h)/det,v=dot(d,q)/det,z=dot(g,q)/det;
        if (z < -1e-10 || u < -1e-11 || v < -1e-11 || u+v>1.+1e-11) continue;
        if (z<1e-10 || u<1e-11 || v<1e-11 || u+v>1.-1e-11) return -1;
        ++hits;
    }
    return hits;
}
extern "C" void skin_enclosure(void* ptr,const double* pts,int count,int* label,double* distance) {
    const auto& mesh=*static_cast<Surface*>(ptr);
    constexpr V directions[]={ {1,.079,.225},{-.073,-.129,1},{.201,-.961,-.226},
                                {-1,-.079,-.225},{.073,.129,-1},{-.201,.961,.226} };
    for(int i=0;i<count;++i) {
        V p={pts[3*i],pts[3*i+1],pts[3*i+2]};int parity=-1,votes=0;bool mismatch=false;
        for(auto d:directions) {
            int h=crossings(mesh,0,p,d);if(h<0)continue;
            int v=h%2;if(votes && v!=parity)mismatch=true;
            parity=v;++votes;
        }
        Match m;mesh.nearest(0,p,m);distance[i]=std::sqrt(m.squared);
        label[i]=votes>=4 && !mismatch && distance[i]>1e-9 ? parity : -1;
    }
}
