#include "Service.hpp" // Include the header for the Service class
#include <cstring> // Include for string manipulation functions
#include <iostream> // Include for input/output stream operations
#include <algorithm> // Include for algorithmic operations

// SetPublisher function: Initializes the list of topics this service will publish
void Service::SetPublisher(std::vector<std::string> topic_names)
{
    publish_list_ = topic_names; // Assign the list of topic names to be published
    for (auto &topic_name : topic_names) // Iterate over each topic name
    {
        topic_buffers_[topic_name] = std::vector<topic_data_t>(buffer_size_); // Initialize the buffer for each topic
        idx_[topic_name] = 0; // Initialize the index for each topic buffer
    }
}

// SetSubscriber function: Initializes the list of topics this service will subscribe to
void Service::SetSubscriber(std::vector<std::string> topic_names)
{
    subscribe_list_ = topic_names; // Assign the list of topic names to be subscribed to
}

// PrepareDataQueue function: Prepares the data queue for each topic with specified sizes
void Service::PrepareDataQueue(std::vector<std::string> topic_names, std::vector<u64> topic_sizes)
{
    for (int i = 0; i < topic_names.size(); i++) // Iterate over each topic name
    {
        std::vector<topic_data_t> topics(buffer_size_); // Create a vector of topic_data_t of size buffer_size_
        for (auto &topic : topics) // Iterate over each topic data structure
        {
            topic.topic_ptr = topic_shared_ptr{new sapeon_byte_t[topic_sizes[i]]}; // Allocate memory for the topic data
            topic.topic_size = topic_sizes[i]; // Set the size of the topic data
        }
        topic_buffers_[topic_names[i]] = topics; // Assign the prepared vector of topics to the topic buffer
    }
}

// createTopicInfo function: Creates and returns a topic_t structure filled with provided information
topic_t Service::createTopicInfo(const std::string topic_name, const u64 start_time, const u32 frame_count)
{
    topic_t topic; // Create an instance of topic_t
    topic.frame_count = frame_count; // Set the frame count for the topic
    topic.set_publish_times(start_time); // Set the publish start time for the topic
    strncpy(topic.service_name, service_name_.c_str(), service_name_.size()+1); // Copy the service name into the topic structure
    strncpy(topic.topic_name, topic_name.c_str(), topic_name.size()+1); // Copy the topic name into the topic structure
    return topic; // Return the filled topic structure
}


// Overloaded Publish function that takes a frame count and a map of topic names to topic data
std::vector<topic_t> Service::Publish(const u32& frame_count, std::map<std::string, topic_data_t> topics)
{
    std::vector<std::string> topic_names; // Vector to store extracted topic names
    std::vector<topic_data_t> topic_datas; // Vector to store extracted topic data
    // Extract topic names and data from the map and populate the vectors
    for(auto& topic : topics){
        topic_names.push_back(topic.first); // Add topic name to the vector
        topic_datas.push_back(topic.second); // Add topic data to the vector
    }
    // Call the main Publish function with the extracted names and data
    return Publish(frame_count, topic_names, topic_datas);;
}

// Overloaded Publish function that takes a frame count, a single topic name, and its data
std::vector<topic_t> Service::Publish(const u32& frame_count, std::string topic_name, topic_data_t topic_data)
{   
    // Call the main Publish function using vectors created from the single topic name and data
    return Publish(frame_count, std::vector<std::string>{topic_name}, std::vector<topic_data_t>{topic_data});
}

// Main Publish function that takes a frame count, a vector of topic names, and a corresponding vector of topic data
std::vector<topic_t> Service::Publish(const u32& frame_count, std::vector<std::string> topic_names, std::vector<topic_data_t> topic_datas)
{
    std::vector<topic_t> results; // Vector to store the topics that will be published
    // Check if the sizes of topic names and topic data vectors match
    if(topic_names.size() != topic_datas.size()){
        throw std::invalid_argument("topic_names and topic_datas size not match"); // Throw an exception if they don't match
    }
    // Iterate over the topic names and data
    for(int i = 0; i < topic_names.size(); i++){
        // Check if the current topic name is in the list of topics to be published
        if(std::find(publish_list_.begin(), publish_list_.end(), topic_names[i]) == publish_list_.end()){
            throw std::invalid_argument("topic_name not in publish_list"); // Throw an exception if it's not in the list
            continue; // Skip the current iteration
        }
        const auto& topic_name = topic_names[i]; // Reference to the current topic name
        const auto& topic_data = topic_datas[i]; // Reference to the current topic data
        // Initialize the index for the topic name if it doesn't exist
        if (idx_.find(topic_name) == idx_.end())
        {
            idx_[topic_name] = 0;
        }
        // Create topic information using the current topic name and frame count
        auto topic = createTopicInfo(topic_name, 0, frame_count);
        topic.topic_data = topic_data; // Set the topic data
        UpdateData(topic_name, topic_data); // Update the data for the current topic
        // Update the index for the current topic, wrapping around if necessary
        idx_[topic_name] = (idx_[topic_name] + 1) % buffer_size_;
        results.push_back(topic); // Add the topic to the results vector
    }
    PublishTopic(results); // Publish the topics
    return results; // Return the vector of published topics
}
// RunService function: Executes the service logic for a subscribed topic and uses a callback to publish results
sapeon_result_t Service::RunService(topic_t subscribed_topic, std::function<sapeon_result_t(std::vector<topic_t> &topics)> publish_callback)
{
    tic_ = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count(); // Record the current time in microseconds
    publish_callback_ = publish_callback; // Store the publish callback function
    auto ret = Run(subscribed_topic); // Execute the main service logic with the subscribed topic
    return ret; // Return the result of the service logic execution
};

// PublishTopic function: Sets the publish time for each topic and invokes the publish callback with the topics
sapeon_result_t Service::PublishTopic(std::vector<topic_t> publish_topics)
{
    for(auto& topic : publish_topics){ // Iterate over each topic to be published
        topic.set_publish_times(tic_); // Set the publish time for the topic
    }
    
    publish_callback_(publish_topics); // Invoke the publish callback with the topics
    return SAPEON_OK; // Return success status
}

// GetData function: Retrieves the data for a given topic name
topic_data_t Service::GetData(std::string topic_name)
{
    if (topic_buffers_.empty()) // Check if the topic buffers map is empty
    {
        return {nullptr, 0}; // Return an empty topic_data_t if there are no buffers
    }
    else if (topic_buffers_.find(topic_name) == topic_buffers_.end()) // Check if the topic name is not in the buffers map
    {
        throw std::invalid_argument("topic_name not in topic_buffers"); // Throw an exception if the topic name is not found
    }
    auto &topic_buffer = topic_buffers_[topic_name]; // Get the buffer for the given topic name
    return topic_buffer[idx_[topic_name]]; // Return the current data for the topic
}

// UpdateData function: Updates the data for a given topic name
sapeon_result_t Service::UpdateData(std::string topic_name, topic_data_t topic_data)
{
    if (topic_buffers_.empty() || topic_buffers_.find(topic_name) == topic_buffers_.end()) // Check if the buffers map is empty or the topic name is not found
    {
        return SAPEON_OK; // Return success status if there's nothing to update
    }
    else
    {
        topic_buffers_[topic_name][idx_[topic_name]] = topic_data; // Update the data for the given topic name
    }
    return SAPEON_OK; // Return success status
}