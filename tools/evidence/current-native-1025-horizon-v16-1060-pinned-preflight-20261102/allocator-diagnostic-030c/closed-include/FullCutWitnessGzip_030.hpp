#pragma once
#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <zlib.h>

namespace full_cut_witness_030 {

inline bool writeGzipText(const std::filesystem::path& path,
                          std::string_view text,
                          std::string& error) {
    gzFile file=gzopen(path.string().c_str(),"wb6");
    if(!file){
        error="gzopen_failed";
        return false;
    }
    std::size_t offset=0;
    while(offset<text.size()){
        const auto amount=std::min<std::size_t>(1u<<20u,text.size()-offset);
        const int written=gzwrite(file,text.data()+offset,static_cast<unsigned int>(amount));
        if(written<=0||static_cast<std::size_t>(written)!=amount){
            int code=Z_OK;
            const char* message=gzerror(file,&code);
            error=std::string("gzwrite_failed:")+(message?message:"unknown");
            (void)gzclose(file);
            return false;
        }
        offset+=amount;
    }
    const int closeStatus=gzclose(file);
    if(closeStatus!=Z_OK){
        error="gzclose_failed:"+std::to_string(closeStatus);
        return false;
    }
    return true;
}

} // namespace full_cut_witness_030
