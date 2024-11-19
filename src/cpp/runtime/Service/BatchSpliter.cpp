#include "BatchSpliter.hpp"

// Function: Run
// Purpose: Splits the subscribed topic's data into smaller batches and publishes them.
// Parameters:
// - subscribed_topic: The topic data that this service has subscribed to and will process.
// Returns: A result code indicating the success or failure of the operation.
sapeon_result_t BatchSpliter::Run(topic_t subscribed_topic){
    // Calculate the size of each topic in the batch
    auto topic_size = subscribed_topic.topic_data.topic_size;
    
    // Check if the topic size is evenly divisible by the number of batches
    if(topic_size % batch_nums_ != 0){
        // If not, throw an exception indicating the configuration error
        throw std::invalid_argument("topic size is not divisible by batch_nums");
    }
    // Calculate the size of each image in the batch
    auto img_size = topic_size / batch_nums_;
    // Prepare a map to store the results of the split operation
    std::map<std::string, topic_data_t> results;
    // Retrieve the initial topic data to use as a template for the split topics
    auto topic_data = GetData(publish_list_[0]);
    // Loop through the number of batches to split the topic data
    for(int i = 0; i < batch_nums_; i++){
        // Calculate the address of the current batch within the original topic data
        auto base_addr = subscribed_topic.topic_data.topic_ptr.get();
        auto img_addr = base_addr + i*img_size;
        // Set the pointer and size for the current batch's topic data
        topic_data.topic_ptr = {img_addr, [](sapeon_byte_t* p){}};
        topic_data.topic_size = img_size;
        // Store the current batch's topic data in the results map
        results[publish_list_[i]] = topic_data;
    }
    
    // Publish the results of the split operation
    Publish(subscribed_topic.frame_count, results);
    // Return a success code
    return SAPEON_OK;
}