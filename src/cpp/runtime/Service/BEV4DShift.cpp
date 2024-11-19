#include "BEV4DShift.hpp"
#include <iostream>

using namespace torch::indexing;
// Function: Run
// Purpose: Processes the subscribed topic by shifting the BEV features based on the provided transformations and rotations.
// Parameters:
// - subscribed_topic: The topic data that this service has subscribed to and will process.
// Returns: A result code indicating the success or failure of the operation.
sapeon_result_t BEV4DShift::Run(topic_t subscribed_topic)
{
  auto topics = service_sync_.GetSync(subscribed_topic, subscribe_list_);

  if (topics.empty() == false)
  {
    auto bev_feature_single = topics["bev_feature_single"];
    auto can = topics["can"];

    parseCanTopic(can);
    auto output = ShiftFeature(inputs, trans_0, rots_0, trans_1, rots_1);
    
    // Update the input tensor with the current BEV features
    torch::Tensor input_cur = torch::zeros({1, height, width, channel_in});
    memcpy(input_cur.data_ptr(), bev_feature_single.topic_data.topic_ptr.get(), sizeof(float) * input_cur.numel()); 
    input_cur = input_cur.permute({0, 3, 1, 2}).contiguous(); 
    inputs = inputs.index({torch::indexing::Slice(0, -1)});
    inputs = torch::concat({input_cur, inputs}); // concat the input tensor with the existing input tensor
    // save the current transformation and rotation data for the next frame
    trans_1 = trans_0;
    rots_1 = rots_0;

    output = output.permute({2, 3, 0, 1}).contiguous(); // Rearrange the dimensions of the output tensor for further processing
    // Add the shifted feature tensor to the outputs queue
    outputs.push(output);
    if (outputs.size() > queue_buffer_size_)
    {                // Ensure the queue does not exceed the buffer size
      outputs.pop(); // Remove the oldest element if the queue is full
    }
    // Prepare the topic data for publishing
    auto topic_data = GetData("bev_feature_multi");
    // The commented-out code below was likely intended for a scenario where dynamic memory allocation is needed
    // if(topic_data.topic_ptr == nullptr){
    //     topic_data.topic_size = channel_in * nframe * height * width * sizeof(float);
    //     topic_data.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_data.topic_size]);
    // }

    topic_data.topic_size = channel_in * nframe * height * width * sizeof(float); // Calculate the size of the topic data
    // Set the topic data pointer to the output tensor's data
    topic_data.topic_ptr = topic_shared_ptr(reinterpret_cast<sapeon_byte_t *>(output.data_ptr()), [](sapeon_byte_t *p) {});

    // The commented-out memcpy and std::cout lines below are likely for debugging or alternative implementation
    // memcpy(topic_data.topic_ptr.get(), output.data_ptr(), topic_data.topic_size);
    // std::cout << subscribed_topic.frame_count << std::endl;
    Publish(subscribed_topic.frame_count, "bev_feature_multi", topic_data); // Publish the shifted BEV features

  }

  return SAPEON_OK; // Return success
}

// Function: ShiftFeature
// Purpose: Performs the feature shifting operation based on the provided transformations and rotations.
// Parameters:
// - input_tensor: The input tensor containing the BEV features to be shifted.
// - trans_0_tensor, rots_0_tensor: The transformation and rotation tensors for the current frame.
// - trans_1_tensor, rots_1_tensor: The transformation and rotation tensors for the adjacent frame.
// - bda_tensor: The tensor containing BDA data.
// Returns: A tensor containing the shifted BEV features.
torch::Tensor BEV4DShift::ShiftFeature(Tensor input_tensor, Tensor trans_0_tensor, Tensor rots_0_tensor, Tensor trans_1_tensor, Tensor rots_1_tensor)
{
  // Prepare transformation matrices for the current and adjacent frames
  auto c02l0 = torch::zeros({nframe, 1, 4, 4});                        // Transformation from current camera frame to current ego frame
  trans_0_tensor = trans_0_tensor.unsqueeze(-1);                       // Add a dimension to the transformation tensor for broadcasting
  c02l0.slice(2, 0, 3).slice(3, 0, 3) = rots_0_tensor.slice(1, 0, 1);  // Set the rotation part of the transformation matrix
  c02l0.slice(2, 0, 3).slice(3, 3, 4) = trans_0_tensor.slice(1, 0, 1); // Set the translation part of the transformation matrix
  c02l0.index_put_({"...", 3, 3}, 1);                                  // Set the homogeneous coordinate to 1

  // - transformation from adjacent camera frame to current ego frame2
  auto c12l0 = torch::zeros({nframe, 1, 4, 4});                        // Initialize a tensor for transformation from adjacent camera frame to current ego frame
  trans_1_tensor = trans_1_tensor.unsqueeze(-1);                       // Add a dimension to the trans_1_tensor for broadcasting
  c12l0.slice(2, 0, 3).slice(3, 0, 3) = rots_1_tensor.slice(1, 0, 1);  // Set the rotation part of the transformation matrix
  c12l0.slice(2, 0, 3).slice(3, 3, 4) = trans_1_tensor.slice(1, 0, 1); // Set the translation part of the transformation matrix
  c12l0.index_put_({"...", 3, 3}, 1);                                  // Set the homogeneous coordinate to 1

  auto c12l0_inv = torch::inverse(c12l0); // Compute the inverse of the transformation matrix
  auto l02l1 = c02l0.matmul(c12l0_inv);   // Compute the transformation matrix from the current ego frame to the adjacent ego frame

  // Mask to select relevant elements for transformation, differing from PyTorch's usual behavior
  auto mask = torch::tensor({{1, 1, 0, 1}, {1, 1, 0, 1}, {0, 0, 0, 0}, {1, 1, 0, 1}}).to(torch::kBool).repeat({nframe, 1, 1}).unsqueeze(1);
  l02l1 = l02l1.masked_select(mask).view({nframe, 1, 1, 3, 3}); // Apply mask and reshape

  // Initialize a tensor for transforming features to BEV space
  auto feat2bev = torch::zeros({3, 3});
  feat2bev.index_put_({0, 0}, grid_interval[0]);                     // Set x-axis scaling factor
  feat2bev.index_put_({1, 1}, grid_interval[1]);                     // Set y-axis scaling factor
  feat2bev.index_put_({0, 2}, grid_lower_bound[0]);                  // Set x-axis translation
  feat2bev.index_put_({1, 2}, grid_lower_bound[1]);                  // Set y-axis translation
  feat2bev.index_put_({2, 2}, 1);                                    // Set homogeneous coordinate
  feat2bev = feat2bev.view({1, 3, 3});                               // Reshape for broadcasting
  auto tf = torch::inverse(feat2bev).matmul(l02l1).matmul(feat2bev); // Compute the final transformation matrix

  // Transform and normalize the grid tensor
  auto grid_tensor = tf.matmul(grid_tensor_ori); // Apply the transformation matrix to the original grid tensor

  // Normalize the grid tensor to the range [-1, 1]
  auto normalize_factor = torch::tensor({width - 1.0, height - 1.0}).view({1, 1, 1, 2}); // Compute normalization factors for width and height
  grid_tensor = grid_tensor.slice(3, 0, 2).slice(4, 0, 1).squeeze();                     // Select and reshape the grid tensor
  grid_tensor = grid_tensor * 2.0 / normalize_factor - 1.0;                              // Normalize to [-1, 1]

  // Set options for grid sampling
  torch::nn::functional::GridSampleFuncOptions options;
  options.align_corners(true);                                   // Align corners of the input and output tensors
  auto out = F::grid_sample(input_tensor, grid_tensor, options); // Apply grid sampling to the input tensor using the normalized grid tensor
  return out;                                                    // Return the transformed and sampled output tensor
}

void BEV4DShift::parseCanTopic(topic_t can)
{
    int32_t offset = 0; // Initialize offset to keep track of the byte position in the subscribed topic's data
    // Copy transformation data for the current frame from the subscribed topic's data to trans_0
    memcpy(trans_0.data_ptr(), can.topic_data.topic_ptr.get(), sizeof(float) * trans_0.numel());
    offset += sizeof(float) * trans_0.numel(); // Update offset after copying

    // Copy rotation data for the current frame from the subscribed topic's data to rots_0
    memcpy(rots_0.data_ptr(), can.topic_data.topic_ptr.get() + offset, sizeof(float) * rots_0.numel());
    offset += sizeof(float) * rots_0.numel(); // Update offset after copying
}
