
#pragma once
#include "common.hpp"
#include "utils.hpp"
#include <Service.hpp>
#include <map>
#include <set>
class ServiceSync{

    public:
    ServiceSync(int32_t buffer_size = 32){
        buffer_size_ = buffer_size;
    };
    std::map<std::string, topic_t> GetSync(topic_t subscribed_topic, const std::vector<std::string>& topic_names);

    private:
    int32_t buffer_size_;
    
    std::map<u32, std::map<std::string, topic_t>> feature_results_;
};