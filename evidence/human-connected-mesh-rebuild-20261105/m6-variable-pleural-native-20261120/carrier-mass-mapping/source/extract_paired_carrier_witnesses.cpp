#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_set>

bool number(std::string_view token,unsigned& result) {
    const auto parsed=std::from_chars(token.data(),token.data()+token.size(),result);
    return parsed.ec==std::errc{} && parsed.ptr==token.data()+token.size();
}

int main(int argc,char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc!=2) return 1;
    std::ifstream wanted(argv[1]);
    std::unordered_set<unsigned> donors,destinations;
    char group;
    unsigned row;
    while (wanted>>group>>row) {
        if (group=='D') donors.insert(row);
        else if (group=='C') destinations.insert(row);
        else return 2;
    }
    if (donors.empty()&&destinations.empty()) return 3;
    // Block-buffered C input avoids one stdio call for each CSV byte on macOS.
    char* buffer = nullptr;
    std::size_t capacity = 0;
    ssize_t length = ::getline(&buffer, &capacity, stdin);
    const auto cleanup = [&buffer](void*) { std::free(buffer); };
    const std::unique_ptr<void, decltype(cleanup)> release(&capacity, cleanup);
    if (length < 0 || !std::string_view(buffer, length).starts_with(
        "record_kind,role,carrier_construction,packet_carrier_row,")) return 4;
    std::array<std::string_view,26> values;
    while ((length = ::getline(&buffer, &capacity, stdin)) >= 0) {
        auto line = std::string_view(buffer, length);
        if (!line.empty() && line.back()=='\n') line.remove_suffix(1);
        if (!line.empty() && line.back()=='\r') line.remove_suffix(1);
        std::string_view remainder(line);
        bool select=false;
        for (std::size_t col=0;col<values.size();++col) {
            const auto delimiter=remainder.find(',');
            if ((col+1<values.size())==(delimiter==std::string_view::npos)) return 5;
            values[col]=remainder.substr(0,delimiter);
            if (col==5) {
                if (!number(values[col],row)) return 6;
                select=donors.contains(row);
            }
            if (col==7) {
                if (!number(values[col],row)) return 7;
                select|=destinations.contains(row);
                if (!select) break;
            }
            if (delimiter!=std::string_view::npos) remainder.remove_prefix(delimiter+1);
        }
        if (select) std::cout<<line<<'\n';
    }
    return std::ferror(stdin) || !std::cout ? 8 : 0;
}
