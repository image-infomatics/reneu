#pragma once

#include <regex>
#include <iterator>
#include <algorithm>

namespace xiuli::utils{

std::regex operator ""_re (char const* const str, std::size_t) {
    return std::regex{str};
}

std::vector<std::string> split(const std::string& text, const std::regex& re) {
    const std::vector<std::string> parts(
        std::sregex_token_iterator(text.begin(), text.end(), re, -1),
        std::sregex_token_iterator());
    return parts;
}

}