#pragma once
#include <tsl/robin_map.h>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>


namespace boost::serialization {
    template<class Archive, class Key, class T>
    void serialize(Archive & ar, tsl::robin_map<Key, T>& map, const unsigned int version) {
        split_free(ar, map, version); 
    }

    template<class Archive, class Key, class T>
    void save(Archive & ar, const tsl::robin_map<Key, T>& map, const unsigned int /*version*/) {
        auto serializer = [&ar](const auto& v) { ar & v; };
        map.serialize(serializer);
    }

    template<class Archive, class Key, class T>
    void load(Archive & ar, tsl::robin_map<Key, T>& map, const unsigned int /*version*/) {
        auto deserializer = [&ar]<typename U>() { U u; ar & u; return u; };
        map = tsl::robin_map<Key, T>::deserialize(deserializer);
    }
} // namespace boost::serialization
