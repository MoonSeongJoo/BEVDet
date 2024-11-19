#include "ServiceSync.hpp"

std::map<std::string, topic_t> ServiceSync::GetSync(topic_t subscribed_topic, const std::vector<std::string>& topic_names)
{
    feature_results_[subscribed_topic.frame_count][subscribed_topic.topic_name] = subscribed_topic;
    
    if(feature_results_.size() > buffer_size_){
        // assert(feature_results_.begin()->first != subscribed_topic.frame_count);
        feature_results_.erase(feature_results_.begin()->first);
    }
    for(auto topic_name : topic_names){
        if(feature_results_[subscribed_topic.frame_count].find(topic_name) == feature_results_[subscribed_topic.frame_count].end()){
            return {};
        }
    }
    auto output = feature_results_[subscribed_topic.frame_count];
    feature_results_.erase(subscribed_topic.frame_count);
    return output;
}