#include "DepthNetPost.hpp" // Include the header file for DepthNetPost class
#include <iostream> // Include for input/output stream operations
#include <cstring> // Include for string manipulation functions
#include <utility> // Include for utility functions like std::move
#include <math.h> // Include for mathematical functions
#include <torch/torch.h> // Include PyTorch C++ API for tensor operations

// Function to run post-processing on depth network output
sapeon_result_t DepthNetPost::Run(topic_t subscribed_topic)
{
    // Check if the topic is related to features
    if (std::string(subscribed_topic.topic_name).find("feature") != std::string::npos)
    {
        // Store the subscribed topic in the depth feature results map
        depth_feat_results_[subscribed_topic.frame_count][subscribed_topic.topic_name] = subscribed_topic;
        // Check if all feature topics for the frame have been received
        if (depth_feat_results_[subscribed_topic.frame_count].size() == depth_net_nums_)
        {
            u32 topic_size = subscribed_topic.topic_data.topic_size; // Get the size of the topic data
            topic_data_t topic_data = GetData("feature"); // Retrieve existing topic data for features
            // Concatenate the current topic data with the existing data
            topic_data = concat(depth_feat_results_[subscribed_topic.frame_count], topic_data, 0, topic_size);
            
            // Publish the concatenated feature data
            Publish(subscribed_topic.frame_count, "feature", topic_data);
        }
    }
    // Check if the topic is related to depth
    else if (std::string(subscribed_topic.topic_name).find("depth") != std::string::npos)
    {
        // Store the subscribed topic in the depth results map
        depth_depth_results_[subscribed_topic.frame_count][subscribed_topic.topic_name] = subscribed_topic;
        // Check if all depth topics for the frame have been received
        if (depth_depth_results_[subscribed_topic.frame_count].size() == depth_net_nums_)
        {
            // Calculate the address offset to skip padded regions in the depth tensor
            u64 addr_offset = depth_tensor_shape[2] * depth_tensor_shape[3] * depth_height_padding * sizeof(float);
            // Calculate the total size of the depth tensor data
            u32 topic_size = depth_tensor_shape[0] * depth_tensor_shape[1] * depth_tensor_shape[2]  * depth_tensor_shape[3] * sizeof(float);
            topic_data_t topic_data = GetData("depth"); // Retrieve existing topic data for depth
            // Concatenate the current depth data with the existing data, considering the address offset
            topic_data = concat_v2(depth_depth_results_[subscribed_topic.frame_count], topic_data, addr_offset, topic_size);
            // Softmax operation is commented out; it was probably intended for normalization
            // softmax(reinterpret_cast<float*>(topic_data.topic_ptr.get()), topic_size * depth_net_nums_ / (depth_tensor_shape[3] * sizeof(float)), depth_tensor_shape[3]);
            depth_depth_results_.erase(subscribed_topic.frame_count); // Clear the depth results for the current frame
            // Publish the concatenated depth data
            Publish(subscribed_topic.frame_count, "depth", topic_data);
        }
    }
    return SAPEON_OK; // Return success status
}

// Function to concatenate version 2 of topic data with preprocessing using PyTorch
topic_data_t DepthNetPost::concat_v2(std::map<std::string, topic_t> topic_outputs, topic_data_t topic_data, u64 addr_offset, u32 topic_size)
{
    // If the topic data pointer is null, allocate memory for it
    if(topic_data.topic_ptr == nullptr){
        // Calculate the total size needed and allocate memory
        topic_data.topic_size = topic_size * depth_net_nums_;
        topic_data.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_data.topic_size]);
    }
    int i = 0; // Index for iterating through topic outputs
    u32 input_size = topic_size; // Size of each input topic data
    // Iterate through each topic output
    for (const auto &s : topic_outputs)
    {
        // Create a tensor from the topic data
        auto options = torch::TensorOptions().dtype(at::kFloat);
        torch::Tensor t = torch::from_blob(s.second.topic_data.topic_ptr.get(), {1, 20, 44, 120}, options);
        // Preprocess the tensor
        t = t.slice(3, 0, 118); // Slice the tensor along the last dimension
        t = t.slice(1, 2, 18); // Slice the tensor along the second dimension
        t = t.transpose(1, 3); // Transpose dimensions 1 and 3
        t = t.transpose(2, 3); // Transpose dimensions 2 and 3
        t = t.softmax(1); // Apply softmax along the first dimension
        // Copy the processed tensor data back into the topic data
        memcpy(topic_data.topic_ptr.get() + input_size * i, t.data_ptr(), topic_size);
        i++; // Move to the next topic output
    }
    return topic_data; // Return the concatenated and processed topic data
}

// Function to concatenate topic data without preprocessing
topic_data_t DepthNetPost::concat(std::map<std::string, topic_t> topic_outputs, topic_data_t topic_data, u64 addr_offset, u32 topic_size)
{
    // If the topic data pointer is null, allocate memory for it
    if(topic_data.topic_ptr == nullptr){
        // Calculate the total size needed and allocate memory
        topic_data.topic_size = topic_size * depth_net_nums_;
        topic_data.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_data.topic_size]);
    }
    int i = 0; // Index for iterating through topic outputs
    u32 input_size = topic_size; // Size of each input topic data
    // Iterate through each topic output
    for (const auto &s : topic_outputs)
    {
        // Copy the topic data into the allocated memory, considering the address offset
        memcpy(topic_data.topic_ptr.get() + input_size * i, s.second.topic_data.topic_ptr.get() + addr_offset, topic_size);
        i++; // Move to the next topic output
    }
    return topic_data; // Return the concatenated topic data
}

// Function to apply softmax normalization to data
void DepthNetPost::softmax(float *data, int32_t vector_nums, int vector_dims)
{
    // Iterate through each vector in the data
    for(int i = 0; i < vector_nums; i++){
        float sum = 0; // Initialize sum for the exponential values of the current vector
        // Compute the exponential of each element in the vector and sum them
        for(int j = 0; j < vector_dims; j++){
            data[i*vector_dims + j] = exp(data[i*vector_dims + j]); // Apply exponential function
            sum += data[i*vector_dims + j]; // Add to sum
        }
        // Normalize each element in the vector by dividing by the sum of exponentials
        for(int j = 0; j < vector_dims; j++){
            data[i*vector_dims + j] /= sum; // Normalize
        }
    }
}