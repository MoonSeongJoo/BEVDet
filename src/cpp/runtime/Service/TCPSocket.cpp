#include "TCPSocket.hpp"

sapeon_result_t TCPSocket::Run(topic_t subscribed_topic)
{
    auto topics = sync_.GetSync(subscribed_topic, subscribe_list_);  // 동기화
    if(topics.empty() == false){
        socket_.send(zmq::message_t(std::string("__start__")), zmq::send_flags::sndmore);  // 데이터 전송
        socket_.send(zmq::message_t(std::to_string(subscribed_topic.frame_count)), zmq::send_flags::none);  // 데이터 전송
        for(auto& topic : topics){
            socket_.send(zmq::message_t(std::string(topic.second.topic_name)), zmq::send_flags::sndmore);  // 데이터 전송
            socket_.send(zmq::message_t(zmq::message_t(topic.second.topic_data.topic_ptr.get(), topic.second.topic_data.topic_size)), zmq::send_flags::none);  // 데이터 전송
        }
    }
    
    return SAPEON_OK;
}