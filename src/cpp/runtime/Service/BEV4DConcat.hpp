#pragma once // Ensures the file is included only once during compilation

#include "common.hpp" // Includes common definitions and utilities
#include "utils.hpp"  // Includes utility functions and classes
#include <Service.hpp> // Includes the base Service class definition
#include <ServiceSync.hpp> // Includes the ServiceSync class for topic synchronization

// Class: BEV4DConcat
// Inherits: Service
// Description: This class is responsible for concatenating single and multi BEV (Bird's Eye View) features
// into a single feature map. This concatenated feature map is then used as input for the BEV encoder.
class BEV4DConcat : public Service
{
public:
    // Constructor
    // Parameters:
    // - service_name: The name of the service, passed to the base class constructor.
    // Initializes the service with a specific service name.
    BEV4DConcat(std::string service_name) : Service(service_name){
    };

    // Function: Run
    // Purpose: Overrides the Run function from the base Service class. It processes the subscribed topic
    // by concatenating BEV features and publishes the result for further processing.
    // Parameters:
    // - subscribed_topic: The topic data that this service has subscribed to and will process.
    // Returns: A result code indicating the success or failure of the operation.
    virtual sapeon_result_t Run(topic_t subscribed_topic);

private:
    ServiceSync service_sync_; // Manages synchronization of topics across different services
    std::map<u32, std::map<std::string, topic_t>> feature_results_; // Stores the results of feature concatenation
    int32_t feature_map_shape_[2] = {128, 128}; // The shape of the feature map (width, height)
    int32_t feature_nums = 80; // The number of features to be concatenated
    int32_t seq_nums = 8; // The number of sequences in the feature map
};