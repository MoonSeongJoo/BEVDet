#pragma once // Ensures the header file is included only once during compilation

#include "common.hpp" // Includes common definitions and utilities
#include "utils.hpp" // Includes utility functions and classes
#include <CMDParser.hpp> // Includes the CMDParser class for command parsing functionality
#include <ServiceSync.hpp> // Includes the ServiceSync class for synchronizing services

// Declares the BEVEncoder class, inheriting from CMDParser
class BEVEncoder: public CMDParser{
public:
    // Declares a virtual method Run that processes a subscribed topic
    virtual sapeon_result_t Run(topic_t subscribed_topic);

    // Constructor for BEVEncoder, initializes the CMDParser base class with provided parameters
    BEVEncoder(x340_rt_api_data_t* api_data, int32_t core_id, std::string service_name, bool dummy_output = false)
    : CMDParser(api_data, core_id, service_name, dummy_output){
    };

    // Declares a virtual method to set input information for DMA operations
    virtual void SetInputInfos(std::vector<dma_file_info_t> input_infos);

    // Declares a virtual method to prepare commands for processing
    virtual bool PrepareCMD();

private:
    // Array to store the shape of the feature map (width and height)
    int32_t feature_map_shape_[2] = {128, 128};
    // Number of features to be processed
    int32_t feature_nums = 80;
    // Number of sequences in the input data
    int32_t seq_nums = 8;
    // Instance of ServiceSync to manage synchronization of services
    ServiceSync service_sync_;
};