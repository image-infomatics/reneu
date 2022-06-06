#pragma once 

// #include "reneu/types.hpp"


namespace reneu::utils{

template<class ARRAY>
void print_array(const ARRAY& arr){
    std::cout<<"print out this array:"<<std::endl;
    for(const auto& x : arr ){
        std::cout<<x<<", ";
    }
    std::cout<<std::endl;
}

}// namespace reneu:utils