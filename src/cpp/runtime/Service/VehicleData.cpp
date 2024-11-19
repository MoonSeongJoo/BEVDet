#include "VehicleData.hpp" // Include the header file for the VehicleData class

// Function to process and publish vehicle data
sapeon_result_t VehicleData::Run(topic_t subscribed_topic)
{
  // Record the start time of the function
  
  torch::jit::script::Module container = torch::jit::load(std::string("/workspace/auto_projects/SAIT_BEV/src/cpp/runtime/build/testset_03_pth/temp/input_data_")+std::to_string(subscribed_topic.frame_count % 38)+std::string(".pth"));

  // Get input image and CAN data tensors from the loaded module
  input = container.attr("imgs").toTensor(); // Image data tensor
  input = input.permute({0, 2, 3, 1}); // Permute the tensor dimensions
  input = input.toType(torch::kFloat); // Convert tensor type to float
  input = input.contiguous(); // Ensure the tensor is stored contiguously in memory

  trans_0 = container.attr("trans").toTensor();
  trans_0.contiguous();
  trans_0 = trans_0.unsqueeze(0);
  rots_0 = container.attr("rots").toTensor();
  rots_0.contiguous();
  rots_0 = rots_0.unsqueeze(0);
    
  auto start_t = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  // Retrieve image data from the subscribed topic
  topic_data_t topic_img = GetData("imgs");
  // Retrieve CAN bus data from the subscribed topic
  topic_data_t topic_can = GetData("can");
  
  // If image data pointer is null, allocate memory and copy the data
  if(topic_img.topic_ptr == nullptr){
    // Calculate the size needed for the image data and allocate memory
    topic_img.topic_size = input.numel() * sizeof(float);
    topic_img.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_img.topic_size]);
  }
  // Copy the image data to the allocated memory
  memcpy(topic_img.topic_ptr.get(), input.data_ptr(), topic_img.topic_size);

  // If CAN data pointer is null, allocate memory and copy the data
  if(topic_can.topic_ptr == nullptr){
    // Calculate the size needed for the CAN data and allocate memory
    topic_can.topic_size = (trans_0.numel() + rots_0.numel())* sizeof(float);
    topic_can.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_can.topic_size]);
  }
  int32_t offset = 0; // Offset for copying data into the allocated memory
  // Copy translation and rotation data for two frames and BDA data to the allocated memory
  memcpy(topic_can.topic_ptr.get(), trans_0.data_ptr(), sizeof(float)*trans_0.numel());
  offset += sizeof(float)*trans_0.numel();
  memcpy(topic_can.topic_ptr.get() + offset, rots_0.data_ptr(), sizeof(float)*rots_0.numel());

  // Record the end time of the function
  auto end_t = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  // Calculate the elapsed time
  auto elapsed_t = end_t - start_t;
  // If the function executed too quickly, wait until the interval is reached
  if(elapsed_t < interval_)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_ - elapsed_t));
  }
  // Publish the processed image and CAN data
  Publish(subscribed_topic.frame_count, "imgs", topic_img);
  Publish(subscribed_topic.frame_count, "can", topic_can);

  return SAPEON_OK; // Return success status
}