#pragma once 
#include "common.hpp"
#include "utils.hpp" 
#include <Service.hpp> 
#include "zmq.hpp" // Includes the ZeroMQ library for messaging
#include "zmq_addon.hpp" // Includes additional utilities for ZeroMQ
#include <ServiceSync.hpp>
class TCPSocket : public Service
{
    public:
    TCPSocket(std::string service_name, std::string port_number = "5555"): Service(service_name), port_number_(port_number){
        ctx_ = zmq::context_t(1);
        socket_ = zmq::socket_t(ctx_, ZMQ_PUSH);
        socket_.connect(std::string("tcp://127.0.0.1:") + port_number);
    }
    virtual sapeon_result_t Run(topic_t subscribed_topic);
private:
    std::string port_number_;
    zmq::context_t ctx_;
    zmq::socket_t socket_;
    ServiceSync sync_;
};