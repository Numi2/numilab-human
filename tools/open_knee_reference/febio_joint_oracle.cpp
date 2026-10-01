// Independent reference driver. Compile against the pinned public FEBio 2.9.0
// sources, not against Matter. Calls the original connector's Update equation.
#include <FECore/FEModel.h>
#include <FECore/FERigidBody.h>
#include <FECore/FETimeInfo.h>
#include <FEBioMech/FERigidCylindricalJoint.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

static vec3d readVector(std::istream& in) { double x,y,z;in>>x>>y>>z;return vec3d(x,y,z); }
static quatd quaternion(std::istream& in) { double x,y,z,w;in>>x>>y>>z>>w;return quatd(x,y,z,w); }
class SourceJoint : public FERigidCylindricalJoint {
public:
    SourceJoint(FEModel* model):FERigidCylindricalJoint(model) {}
    void configure(std::istream& in,FERigidBody& a,FERigidBody& b) {
        a.m_r0=readVector(in);b.m_r0=readVector(in);a.m_rt=readVector(in);b.m_rt=readVector(in);
        m_q0=readVector(in);vec3d axis=readVector(in);
        a.SetRotation(quaternion(in));b.SetRotation(quaternion(in));
        a.m_rp=a.m_rt;b.m_rp=b.m_rt;a.m_qp=a.GetRotation();b.m_qp=b.GetRotation();
        m_L=readVector(in);m_U=readVector(in);
        in>>m_eps>>m_ups>>m_dp>>m_qp>>m_Fp>>m_Mp;
        in>>m_bd>>m_bq;
        m_rbA=&a;m_rbB=&b;m_qa0=m_q0-a.m_r0;m_qb0=m_q0-b.m_r0;
        m_ea0[0]=m_eb0[0]=axis;
        // Transverse directions have no contribution in this alpha=1 source
        // equation, but initialize them rather than read indeterminate values.
        m_ea0[1]=m_ea0[2]=m_eb0[1]=m_eb0[2]=vec3d(0,0,0);
    }
};
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    std::ifstream in(argv[1]);std::ofstream out(argv[2]);size_t count=0;in>>count;
    if(!in||!out||!count||count>100000)return 2;
    out<<std::setprecision(17);
    FEModel model;FERigidBody a(&model),b(&model);SourceJoint joint(&model);
    FETimeInfo time;time.alpha=1;
    for(size_t i=0;i<count;++i) {
        joint.configure(in,a,b);if(!in)return 2;
        joint.Update(0,time);
        out<<joint.m_F.x<<' '<<joint.m_F.y<<' '<<joint.m_F.z<<' '
           <<joint.m_M.x<<' '<<joint.m_M.y<<' '<<joint.m_M.z<<'\n';
    }
    return out?0:2;
}
