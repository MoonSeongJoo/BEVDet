#pragma once // Ensures the header file is included only once in a single compilation

#include <common.hpp> // Includes the common definitions and utilities
#include <Core/Service.hpp> // Includes the Service class from the Core module
#include "zmq.hpp" // Includes the ZeroMQ library for messaging
#include "zmq_addon.hpp" // Includes additional utilities for ZeroMQ
#include <mutex> // Includes the mutex class for thread synchronization

// Declares the ServiceWorker class
class ServiceWorker{
public:
    // Constructor for ServiceWorker
    ServiceWorker(zmq::socket_ref publisher, zmq::context_t *ctx, std::shared_ptr<Service> service, std::mutex& mtx)
    :publisher_(publisher), ctx_(ctx), service_(service), mtx_(mtx), tic_(0){
        // Initializes the subscriber socket and connects it to the "inproc://topics" endpoint
        subscriber_ = std::make_shared<zmq::socket_t>(*ctx_, zmq::socket_type::sub);
        subscriber_->connect(std::string("inproc://topics"));
        // Sets the receive high water mark for the subscriber socket
        subscriber_->set(zmq::sockopt::rcvhwm, 1024*1024);
        
        // Subscribes to topics provided by the service
        for (const auto &subscribe_topic : service_->GetSubscribeTopicNames())
        {
            subscriber_->set(zmq::sockopt::subscribe, subscribe_topic);
        }
        // Subscribes to the "__status__" topic for control messages
        subscriber_->set(zmq::sockopt::subscribe, "__status__");

    };

    // Enables the repeat functionality to republish messages
    void EnableRepeat(){
        repeat_ = true;
    }
    // Disables the repeat functionality
    void DisableRepeat(){
        repeat_ = false;
    }
    
    // Declares the DoWork function that processes messages
    sapeon_result_t DoWork();

    // Declares the Publish function to publish topics
    sapeon_result_t Publish(std::vector<topic_t>& topics);
private:

    std::mutex& mtx_; // Reference to a mutex for thread synchronization
    zmq::context_t* ctx_; // Pointer to the ZeroMQ context
    std::shared_ptr<Service> service_; // Shared pointer to the Service instance
    zmq::socket_ref publisher_; // Reference to the publisher socket
    std::shared_ptr<zmq::socket_t> subscriber_; // Shared pointer to the subscriber socket
    bool repeat_ = false; // Flag to control the repeat functionality
    u64 tic_; // Timestamp variable (unused in the provided code)
};