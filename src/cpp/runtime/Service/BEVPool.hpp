#pragma once
#include "common.hpp" // Common includes and definitions for the project
#include "utils.hpp" // Utility functions and classes
#include <Service.hpp> // Base class for services
#include <Eigen/Dense> // Linear algebra library
#include <unsupported/Eigen/CXX11/Tensor> // Tensor operations for Eigen
#include <ServiceSync.hpp> // Synchronization utilities for services

// BEVPool class inherits from Service and provides functionality for Bird's Eye View pooling
class BEVPool : public Service
{
    // Type alias for the output of voxel pooling operation
    using voxel_pool_output_t = std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>;
public:
    // Constructor that initializes the BEVPool service
    BEVPool(std::string service_name) : Service(service_name)
    {
        // Initialization of parameters for frustum creation and data loading
        const float lower_bound = 1.0, upper_bound = 60.0, interval = 0.5;  
        const float input_height = 256.0, input_width = 704.0, downsample = 16.0;
        // Creates a frustum and determines its shape based on the given parameters
        auto [frustum, shape] = createFrustum(lower_bound, upper_bound, interval, input_height, input_width, downsample);
        // Path to the data files
        const std::string data_root = "./bev_pool/";
        // Loading transformation matrices and calibration data from binary files
        auto rots = readBinaryFile<float>(data_root + "rots_1_6_3_3.bin");
        auto trans = readBinaryFile<float>(data_root + "trans_1_6_3.bin");
        auto cam2imgs = readBinaryFile<float>(data_root + "cam2imgs_1_6_3_3.bin");
        auto post_rots = readBinaryFile<float>(data_root + "post_rots_1_6_3_3.bin");
        auto post_trans = readBinaryFile<float>(data_root + "post_trans_1_6_3.bin");
        auto bda = readBinaryFile<float>(data_root + "bda_1_3_3.bin");
        // Computes the coordinates of LiDAR points in the camera frame
        auto coor = getLidarCoor(1, 6, frustum, shape, rots, trans, cam2imgs, post_rots, post_trans, bda);
        // Prepares data for voxel pooling operation
        std::tie(ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths) = voxelPoolingPrepareV2(1, 6, coor, shape, {-51.2, -51.2, -5.0}, {0.8, 0.8, 8.0}, {128, 128, 1});
    };
    // Virtual function to be implemented for processing subscribed topics
    virtual sapeon_result_t Run(topic_t subscribed_topic);

private:
    // Kernel function for BEV pooling operation
    void bevPoolV2Kernel(int c, int n_intervals, const float *depth, const float *feat, const size_t *ranks_depth, const size_t *ranks_feat, const size_t *ranks_bev, const size_t *interval_starts, const size_t *interval_lengths, float *out);
    // Improved kernel function for BEV pooling operation
    void BevPoolV2KernelImproved(int c, int n_intervals, const float *depth, const float *feat, const size_t *ranks_depth, const size_t *ranks_feat, const size_t *ranks_bev, const size_t *interval_starts, const size_t *interval_lengths, float *out);
    // Wrapper function to prepare and call the BEV pooling kernel function
    void bevPoolV2Forward(const float* _depth, const std::pair<const float* , int> _feat, float* _out, const std::vector<size_t> &_ranks_depth, const std::vector<size_t> &_ranks_feat, const std::vector<size_t> &_ranks_bev, const std::vector<size_t> &_interval_lengths, const std::vector<size_t> &_interval_starts);
    // Function to compute LiDAR coordinates in the camera frame
    std::vector<float> getLidarCoor(size_t B, size_t N, const std::vector<float> &frustum, const std::vector<size_t> &shape, const std::vector<float> &rots, const std::vector<float> &trans, const std::vector<float> &cam2imgs, const std::vector<float> &post_rots, const std::vector<float> &post_trans, const std::vector<float> &bda);
    // Function to create a frustum and determine its shape
    std::pair<std::vector<float>, std::vector<size_t>> createFrustum(float lower_bound, float upper_bound, float interval, float input_height, float input_width, float downsample);
    // Function to prepare data for voxel pooling operation
    voxel_pool_output_t voxelPoolingPrepareV2(size_t B, size_t N, const std::vector<float> &coor, const std::vector<size_t> &shape, const std::vector<float> &grid_lower_bound, const std::vector<float> &grid_interval, const std::vector<int> &grid_size);

    // Synchronization utility for services
    ServiceSync service_sync_;
    // Shape of the feature map
    int32_t feature_map_shape_[2] = {128, 128};
    // Number of features to be processed
    int32_t feature_nums_ = 80;

    // Ranks for BEV, depth, and feature data used in pooling operations
    std::vector<size_t> ranks_depth;
    std::vector<size_t> ranks_feat;
    std::vector<size_t> ranks_bev;
    std::vector<size_t> interval_starts;
    std::vector<size_t> interval_lengths;
};
