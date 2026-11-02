#include "FullCutWitnessGzip_030.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include <zlib.h>

namespace {
std::string fixture() {
    std::string text="shared_capacity_full_cut_begin encoding=hex_lsb_first_index_order_parent_then_child parent_mask_hex=abc child_mask_hex=def\n";
    for (int i=0;i<50000;++i) {
        text+=std::to_string(i);
        text+=",15,69,8123,0.000123456789,group,face;face2\n";
    }
    text.push_back('\0');
    text.push_back(static_cast<char>(0xff));
    return text;
}
bool readGzip(const std::filesystem::path& path,std::string& output) {
    gzFile file=gzopen(path.string().c_str(),"rb");
    if(!file)return false;
    char chunk[65536];
    for(;;){
        const int n=gzread(file,chunk,sizeof(chunk));
        if(n>0){output.append(chunk,static_cast<std::size_t>(n));continue;}
        int code=Z_OK;
        (void)gzerror(file,&code);
        const int closeStatus=gzclose(file);
        return n==0&&code==Z_OK&&closeStatus==Z_OK;
    }
}
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    const auto dir=std::filesystem::path(argv[1]);
    std::filesystem::create_directories(dir);
    const std::string original=fixture();
    std::string error;
    const auto path=dir/"full-cut-witness.csv.gz";
    if(!full_cut_witness_030::writeGzipText(path,original,error)){
        std::cerr<<error<<"\n";return 3;
    }
    std::string restored;
    if(!readGzip(path,restored)||restored!=original)return 4;
    const auto emptyPath=dir/"empty.gz";
    if(!full_cut_witness_030::writeGzipText(emptyPath,std::string_view{},error))return 5;
    std::string empty;
    if(!readGzip(emptyPath,empty)||!empty.empty())return 6;
    std::cout<<"PASS exact_bytes="<<original.size()<<" gzip_bytes="
             <<std::filesystem::file_size(path)<<" empty_gzip=PASS\n";
    return 0;
}
