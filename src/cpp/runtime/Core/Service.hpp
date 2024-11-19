#pragma once // Ensures the header file is included only once in a single compilation
#include <string> // Includes the string library for string manipulation
#include <map> // Includes the map container for mapping keys to values
#include <queue> // Includes the queue container for FIFO data structure
#include <vector> // Includes the vector container for dynamic arrays
#include <memory> // Includes the memory library for dynamic memory management
#include <common.hpp> // Includes common definitions and utilities
#include <mutex> // Includes the mutex library for managing locks
class ServiceManager; // Forward declaration of the ServiceManager class

class Service {
public:
    // Constructor: Initializes a Service object with a given service name
    Service(std::string service_name){
        service_name_ = service_name; // Assigns the provided service name to the service_name_ member variable
    };
    
    ~Service(){}; // Destructor: Cleans up resources used by the Service object

    // SetPublisher: Sets the list of topics this service will publish
    void SetPublisher(std::vector<std::string> topic_names);
    // SetSubscriber: Sets the list of topics this service will subscribe to
    void SetSubscriber(std::vector<std::string> topic_names);
    // PrepareDataQueue: Prepares the data queue for each topic with specified sizes
    void PrepareDataQueue(std::vector<std::string> topic_names, std::vector<u64> topic_sizes);
    
    // GetServiceName: Returns the name of the service
    std::string GetServiceName(){
        return service_name_;
    }
    // GetSubscribeTopicNames: Returns the list of topics this service subscribes to
    std::vector<std::string> GetSubscribeTopicNames(){
        return subscribe_list_;
    };
    // GetPublishTopicNames: Returns the list of topics this service publishes
    std::vector<std::string> GetPublishTopicNames(){
        return publish_list_;
    }

    // Run: Pure virtual function to be implemented by derived classes for service logic
    virtual sapeon_result_t Run(topic_t subscribed_topic) = 0;
    
    // RunService: Executes the service logic for a subscribed topic and uses a callback to publish results
    sapeon_result_t RunService(topic_t subscribed_topic, std::function<sapeon_result_t(std::vector<topic_t>& topics)> publish_callback = nullptr);
    // PublishTopic: Publishes a list of topics
    sapeon_result_t PublishTopic(std::vector<topic_t> publish_topics);

    // GetData: Retrieves the data for a given topic name
    topic_data_t GetData(std::string topic_name);
    // UpdateData: Updates the data for a given topic name
    sapeon_result_t UpdateData(std::string topic_name, topic_data_t topic_data);

protected:
    // createTopicInfo: Creates and returns a topic_t structure filled with provided information
    topic_t createTopicInfo(const std::string topic_name, const u64 start_time = 0, const u32 frame_count = 0);
    // Publish: Overloaded methods to publish data for a given frame count and set of topics
    std::vector<topic_t> Publish(const u32& frame_count, std::vector<std::string> topic_names, std::vector<topic_data_t> topic_datas);
    std::vector<topic_t> Publish(const u32& frame_count, std::string topic_name, topic_data_t topic_data);
    std::vector<topic_t> Publish(const u32& frame_count, std::map<std::string, topic_data_t> topics);
    std::vector<std::string> publish_list_; // List of topics this service will publish
    std::vector<std::string> subscribe_list_; // List of topics this service will subscribe to
    std::string service_name_; // Name of the service

private:
    std::map<std::string, std::vector<topic_data_t>> topic_buffers_; // Buffers for storing topic data
    const u32 buffer_size_ = 32; // Default size for topic data buffers
    std::map<std::string, u32> idx_; // Current index for each topic buffer
    u64 tic_; // Timestamp for the last operation
    std::function<sapeon_result_t(std::vector<topic_t>& topics)> publish_callback_; // Callback function for publishing topics
};