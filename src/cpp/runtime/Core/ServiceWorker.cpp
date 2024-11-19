#include <Core/ServiceWorker.hpp> // Includes the header for the ServiceWorker class from the Core directory
#include "ServiceWorker.hpp" // Redundant include, possibly a mistake

// Function DoWork: Main loop for the ServiceWorker to process messages
sapeon_result_t ServiceWorker::DoWork()
{
  while (true) // Infinite loop to continuously process incoming messages
  {
    std::vector<zmq::message_t> recv_msgs; // Vector to hold received messages
    zmq::recv_result_t ret = zmq::recv_multipart(*subscriber_, std::back_inserter(recv_msgs)); // Receives multipart message
    assert(ret && "recv failed"); // Asserts that the receive operation did not fail
    assert(*ret == 2); // Asserts that exactly two parts were received (topic name and data)
    auto recv_topic_name = recv_msgs[0].to_string(); // Converts the first message part to a string as the topic name
    if (recv_topic_name == "__status__") // Checks if the message is a status message
    {
      auto status = recv_msgs[1].to_string(); // Converts the second message part to a string as the status
      if (status == "exit") // Checks if the status is an exit command
      {
        break; // Exits the loop (and thus the function) if status is "exit"
      }
    }

    for (auto &subscribe_topic : service_->GetSubscribeTopicNames()) // Iterates over subscribed topics
    {
      mtx_.lock(); // Locks the mutex to protect shared resources
      // Debugging line commented out: prints service name, subscribed topic, and received topic name
      mtx_.unlock(); // Unlocks the mutex
      if (recv_topic_name == subscribe_topic || subscribe_topic == "") // Checks if the received topic is subscribed to or if the subscription is to all topics
      {
        auto recv_topic = reinterpret_cast<topic_t *>(recv_msgs[1].data()); // Interprets the second message part as topic data
        mtx_.lock(); // Locks the mutex again for shared resource protection
        // Debugging line commented out: prints service name, topic name, and frame count
        if(recv_topic->topic_data.topic_ptr != nullptr){
          // std::cout << "Received topic: " << recv_topic_name << " use count: " << recv_topic->frame_count << std::endl;
          // Debugging line commented out: prints service name, topic name, and use count of the topic pointer
        }
        mtx_.unlock(); // Unlocks the mutex
        service_->RunService(*recv_topic, std::bind(&ServiceWorker::Publish, this, std::placeholders::_1)); // Processes the received topic
        if(repeat_){ // Checks if the topic should be published again (repeated)
          mtx_.lock(); // Locks the mutex to protect shared resources during repeat publish
          auto start_t = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count(); // Gets the current time in microseconds
          recv_topic->set_publish_times(start_t); // Sets the publish time for the topic
          recv_topic->frame_count += 1; // Increments the frame count for the topic
          publisher_.send(zmq::message_t(std::string(recv_topic_name)), zmq::send_flags::sndmore); // Sends the topic name as the first part of the message
          publisher_.send(zmq::message_t(recv_topic, sizeof(topic_t)), zmq::send_flags::none); // Sends the topic data as the second part of the message
          mtx_.unlock(); // Unlocks the mutex
        }
        // This line sets the publish time for the received topic using the start time provided.
        // topic.set_publish_times(tic_);
        // break;
      }
    }
  }
  return SAPEON_OK;
}

// Function Publish: Publishes a list of topics
sapeon_result_t ServiceWorker::Publish(std::vector<topic_t>& topics)
{
  for (auto &topic : topics) // Iterates over each topic in the list
  {
    // Locks the mutex to ensure thread-safe operations when publishing topics
    mtx_.lock();
    // Sends the topic name as the first part of a multipart message
    publisher_.send(zmq::message_t(std::string(topic.topic_name)), zmq::send_flags::sndmore);
    // Sends the topic data as the second part of the multipart message
    publisher_.send(zmq::message_t(&topic, sizeof(topic_t)), zmq::send_flags::none);
    // Unlocks the mutex after publishing the topic
    mtx_.unlock();
  }
  // Returns SAPEON_OK indicating successful publication of all topics
  return SAPEON_OK;
}