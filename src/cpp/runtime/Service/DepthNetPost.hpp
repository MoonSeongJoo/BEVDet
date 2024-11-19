#pragma once // Ensures the file is included only once during compilation
#include "common.hpp" // Includes common definitions and utilities
#include "utils.hpp" // Includes utility functions and classes
#include <Service.hpp> // Includes the base Service class definition

// DepthNetPost class inherits from Service and is used for post-processing depth network outputs
class DepthNetPost : public Service
{
public:
    // Constructor initializing DepthNetPost with the number of depth networks and service name
    DepthNetPost(int32_t depth_net_nums, std::string service_name) : Service(service_name), depth_net_nums_(depth_net_nums){
    };
    // Virtual function to run the post-processing on subscribed topic data
    virtual sapeon_result_t Run(topic_t subscribed_topic);

private:
    // Function to concatenate version 2 of topic data with preprocessing
    topic_data_t concat_v2(std::map<std::string, topic_t> topic_outputs, topic_data_t topic_data, u64 addr_offset, u32 topic_size);
    // Function to concatenate topic data without preprocessing
    topic_data_t concat(std::map<std::string, topic_t> topic_outputs, topic_data_t topic_data, u64 addr_offset, u32 topic_size);
    // Function to apply softmax normalization to data
    void softmax(float* data, int32_t vector_nums, int vector_dims);
    int32_t depth_net_nums_; // Number of depth networks used for processing
    // Maps to store depth results for each frame, categorized by depth and feature data
    std::map<u32, std::map<std::string, topic_t>> depth_depth_results_;
    std::map<u32, std::map<std::string, topic_t>> depth_feat_results_;
    int32_t depth_tensor_shape[4] = {1, 16, 44, 118}; // Shape of the depth tensor

    // Constant for the depth height padding, used for aspp layer compile in depthnet model
    const int32_t depth_height_padding = 2;
};