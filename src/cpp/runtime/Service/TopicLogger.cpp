#include "TopicLogger.hpp" // Include the header file for the TopicLogger class
#include <iostream> // Include for input/output stream operations
#include <fstream> // Include for file stream operations

// Run function processes and logs the subscribed topic data
sapeon_result_t TopicLogger::Run(topic_t subscribed_topic)
{
    json topic; // Create a JSON object to store topic data
    // Populate the JSON object with topic details
    topic["args"]["service_name"] = subscribed_topic.service_name;
    topic["args"]["topic_name"] = subscribed_topic.topic_name;
    topic["dur"] = subscribed_topic.elapsed_time; // Duration of the topic processing
    topic["ts"] = subscribed_topic.publish_time - subscribed_topic.elapsed_time; // Timestamp of the topic
    topic["tid"] = subscribed_topic.topic_name; // Topic ID
    topic["pid"] = subscribed_topic.service_name; // Process ID, using service name
    topic["ph"] = "X"; // Phase of the event, marked as 'X' (unknown or custom)
    topic["name"] = std::to_string(subscribed_topic.frame_count); // Name of the event, using frame count
    log_data_["traceEvents"].push_back(topic); // Add the topic JSON object to the log data

    if (topic_dump_flag_) // Check if topic dumping is enabled
    {
        Dump(subscribed_topic); // Call the Dump function to save topic data to a file
    }
    return SAPEON_OK; // Return success status
}

// Dump function saves the subscribed topic data to a binary file
void TopicLogger::Dump(topic_t subscribed_topic)
{
    std::string root_path = "./dump/"; // Root path for dumping files
    // Construct the file name using service name, topic name, and frame count
    std::string file_name = std::string(subscribed_topic.service_name) + std::string("_") + std::string(subscribed_topic.topic_name) + std::string("_") + std::to_string(subscribed_topic.frame_count) + ".bin";
    std::string file_path = root_path + file_name; // Full path to the dump file
    std::ofstream f(file_path, std::ios::binary); // Open a file stream in binary mode
    // Write the topic data to the file
    f.write((char*)subscribed_topic.topic_data.topic_ptr.get(), subscribed_topic.topic_data.topic_size);
}