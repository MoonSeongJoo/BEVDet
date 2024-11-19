#pragma once // Ensures the header is included only once by the preprocessor

#include "common.hpp" // Includes common definitions and utilities
#include "utils.hpp" // Includes utility functions and classes
#include <Service.hpp> // Includes the base Service class for creating services
#include <torch/torch.h> // Includes the PyTorch C++ API for tensor operations
#include <torch/script.h> // Includes support for loading TorchScript modules

// VehicleData class inherits from Service and is designed to process vehicle data
class VehicleData : public Service
{
public:
    // Constructor that initializes the VehicleData service with a service name
    VehicleData(std::string service_name) : Service(service_name) {
        // Load a TorchScript module containing the vehicle data
        torch::jit::script::Module container = torch::jit::load("input.pth");
        // Extract and preprocess tensors from the loaded module
        input = container.attr("imgs").toTensor().slice(0, 0, 1); // Image data tensor
        input.contiguous();
        trans_0 = container.attr("trans").toTensor().slice(0, 0, 1); // Current translation data tensor
        trans_0.contiguous();
        rots_0 = container.attr("rots").toTensor().slice(0, 0, 1); // Current rotation data tensor
        rots_0.contiguous();
        trans_1 = container.attr("trans_prev").toTensor().slice(0, 0, 1); // Previous translation data tensor
        trans_1.contiguous();
        rots_1 = container.attr("rots_prev").toTensor().slice(0, 0, 1); // Previous rotation data tensor
        rots_1.contiguous();
        bda = container.attr("bda").toTensor().slice(0, 0, 1); // Behavioral Driver Assistance (BDA) data tensor

        // Further preprocessing on the image tensor for compatibility
        input = input.index({0}); // Select the first element in the batch
        input = input.permute({0, 2, 3, 1}); // Permute the tensor dimensions
        input = input.toType(torch::kFloat); // Convert tensor type to float
        input = input.contiguous(); // Ensure the tensor is stored contiguously in memory
    }
    // Virtual function to be overridden that processes and publishes vehicle data
    virtual sapeon_result_t Run(topic_t subscribed_topic);
    
private:
    // Private member variables to store processed tensors
    torch::Tensor input; // Tensor for image data
    torch::Tensor trans_0; // Tensor for current translation data
    torch::Tensor trans_1; // Tensor for previous translation data
    torch::Tensor rots_0; // Tensor for current rotation data
    torch::Tensor rots_1; // Tensor for previous rotation data
    torch::Tensor bda; // Tensor for BDA data
    u64 interval_ = 50; // Interval in milliseconds to control data processing rate
};