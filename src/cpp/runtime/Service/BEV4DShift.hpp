#pragma once
#include "common.hpp"     // Include common definitions and utilities
#include "utils.hpp"      // Include utility functions
#include <Service.hpp>    // Include the Service base class
#include <torch/torch.h>  // Include the PyTorch C++ API
#include <torch/script.h> // Include the header file for torch::jit::load
#include <ServiceSync.hpp>
using namespace torch;               // Use the torch namespace for convenience
using namespace std;                 // Use the std namespace for convenience
namespace F = torch::nn::functional; // Alias the torch::nn::functional namespace for convenience

// BEV4DShift class inherits from the Service class
class BEV4DShift : public Service
{
public:
    // Constructor that takes a service name and initializes the Service base class with it
    BEV4DShift(std::string service_name) : Service(service_name)
    {
        // Load a PyTorch script module from a file
        torch::jit::script::Module container = torch::jit::load("input.pth");
        // Extract the "feat_prev" tensor from the module, slice it, and tile it to match the desired shape
        inputs = container.attr("feat_prev").toTensor().slice(0, 0, 1);
        inputs = inputs.tile({8, 1, 1, 1});

        // Define tensor options for creating tensors with float data type
        auto options = torch::TensorOptions().dtype(at::kFloat);

        // inputs = torch::zeros({8, 80, 128, 128}, options);
        // Initialize transformation and rotation tensors for time step 0
        trans_0 = torch::zeros({1, 6, 3}, options);
        rots_0 = torch::zeros({1, 6, 3, 3}, options);
        // Initialize transformation and rotation tensors for time ostep 1
        trans_1 = torch::zeros({1, 6, 3}, options);
        rots_1 = torch::zeros({1, 6, 3, 3}, options);

        trans_0 = container.attr("trans").toTensor().slice(0, 0, 1); // Current translation data tensor
        rots_0 = container.attr("rots").toTensor().slice(0, 0, 1); // Current rotation data tensor
        trans_1 = container.attr("trans_prev").toTensor().slice(0, 0, 1); // Previous translation data tensor
        rots_1 = container.attr("rots_prev").toTensor().slice(0, 0, 1); // Previous rotation data tensor

        // Define the lower bound and interval for the grid, as well as its size
        grid_lower_bound = torch::tensor({-51.2, -51.2, -5.0});
        grid_interval = torch::tensor({0.8, 0.8, 8.0});
        grid_size = torch::tensor({128.0, 128.0, 1.0});
        // Generate a grid tensor based on the image dimensions
        xs = torch::linspace(0, width - 1, width).view({1, width}).expand({height, width});
        ys = torch::linspace(0, height - 1, height).view({height, 1}).expand({height, width});
        xs_ones_like = torch::ones_like(xs);
        // Stack the xs, ys, and ones tensors to form the original grid tensor
        grid_tensor_ori = torch::stack({xs, ys, xs_ones_like}, -1);
        // Expand and reshape the grid tensor to match the number of frames and image dimensions
        grid_tensor_ori = grid_tensor_ori.view({1, height, width, 3}).expand({nframe, height, width, 3}).view({nframe, height, width, 3, 1});
    };

    // Virtual function to be overridden for processing subscribed topics
    virtual sapeon_result_t Run(topic_t subscribed_topic);

private:
    // Function to shift feature maps based on transformations and rotations
    torch::Tensor ShiftFeature(Tensor input_tensor, Tensor trans_0_tensor, Tensor rots_0_tensor, Tensor trans_1_tensor, Tensor rots_1_tensor);

    void parseCanTopic(topic_t can);
    // Tensor to store the lower bound of the grid in 3D space
    torch::Tensor grid_lower_bound;
    // Tensor to store the interval of the grid in 3D space
    torch::Tensor grid_interval;
    // Tensor to store the size of the grid in 3D space
    torch::Tensor grid_size;

    // Number of frames to process
    int nframe = 8;
    // Number of input channels
    int channel_in = 80;
    // Height of the input image
    int height = 128;
    // Width of the input image
    int width = 128;

    // Tensors for x and y coordinates and a tensor of ones for homogeneous coordinates
    torch::Tensor xs;
    torch::Tensor ys;
    torch::Tensor xs_ones_like;
    // Original grid tensor before any transformations
    torch::Tensor grid_tensor_ori;

    // Tensor for input features
    torch::Tensor inputs;
    // Tensors for transformations and rotations at time step 0
    torch::Tensor trans_0;
    torch::Tensor rots_0;
    // Tensors for transformations and rotations at time step 1
    torch::Tensor trans_1;
    torch::Tensor rots_1;

    // Queue to store output tensors
    std::queue<torch::Tensor> outputs;
    // Buffer size for the output queue
    int32_t queue_buffer_size_ = 32;
    ServiceSync service_sync_;
};