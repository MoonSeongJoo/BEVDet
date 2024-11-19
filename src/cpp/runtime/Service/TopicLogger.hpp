#pragma once // Ensures the file is included only once during compilation
#include "common.hpp" // Includes common definitions and utilities
#include "utils.hpp" // Includes utility functions and classes
#include <Service.hpp> // Includes the base Service class definition
#include "json.hpp" // Includes the JSON library for handling JSON data
#include <fstream> // Includes the file stream library for file operations

using json = nlohmann::json; // Defines a convenient alias for the JSON library namespace

// TopicLogger class inherits from Service and is specialized for logging topic data
class TopicLogger : public Service
{
public:
    // Constructor for TopicLogger, initializes logging to a specified file path
    TopicLogger(std::string log_file_path, std::string service_name) : Service(service_name)
    {
        subscribe_list_.push_back(""); // Initializes the subscription list (empty means subscribe to all)
        log_file_path_ = log_file_path; // Stores the log file path
        log_file_ = std::ofstream(log_file_path_); // Opens the log file for writing
        log_data_["displayTimeUnit"] = "ms"; // Sets the time unit for display in the log file
        log_data_["traceEvents"] = json::array(); // Initializes the trace events as a JSON array
    };
    // Destructor for TopicLogger, writes the log data to file and closes the file
    ~TopicLogger()
    {
        log_file_ << log_data_.dump(4); // Dumps the log data to the file with indentation
        log_file_.close(); // Closes the log file
    };
    // Virtual function to process and log the subscribed topic data
    virtual sapeon_result_t Run(topic_t subscribed_topic);
    // Function to dump the subscribed topic data to a binary file
    void Dump(topic_t subscribed_topic);

private:
    json log_data_; // JSON object to store log data
    std::string log_file_path_; // Path to the log file
    std::ofstream log_file_; // File stream for the log file
    bool topic_dump_flag_ = false; // Flag to control dumping of topic data to files
};