// Guard to prevent multiple inclusions of the header
#pragma once

// Include necessary headers
#include "common.hpp" // Common definitions and utilities
#include "utils.hpp"  // Utility functions and classes
#include <Service.hpp> // Base class Service definition

// Class: BatchSpliter
// Inherits from: Service
// Description: This class is designed to split incoming data into smaller batches for processing.
// It is particularly useful in scenarios where handling large volumes of data at once is impractical or inefficient.
class BatchSpliter : public Service
{
public:
    // Constructor
    // Parameters:
    // - batch_nums: The number of batches to split the data into.
    // - service_name: The name of the service, passed to the base class constructor.
    // Initializes the service with a specific service name and sets the number of batches.
    BatchSpliter(int32_t batch_nums, std::string service_name) : Service(service_name), batch_nums_(batch_nums){
    };

    // Function: Run
    // Purpose: Processes the subscribed topic and splits its data into smaller batches.
    // Parameters:
    // - subscribed_topic: The topic data that this service has subscribed to and will process.
    // Returns: A result code indicating the success or failure of the operation.
    virtual sapeon_result_t Run(topic_t subscribed_topic);

private:
    // Member: batch_nums_
    // Type: int32_t
    // Description: Stores the number of batches to split the data into.
    int32_t batch_nums_;
};