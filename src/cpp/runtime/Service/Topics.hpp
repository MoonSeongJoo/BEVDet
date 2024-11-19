#pragma once
#include "common.hpp"
#include <vector>
#include <chrono>
#include <memory>
#include <api/c/x340_rt_grpc.h>
#include <torch/torch.h>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

using topic_shared_ptr = std::shared_ptr<sapeon_byte_t[]>;


struct motion_data_t{
  torch::Tensor trans_0;
  torch::Tensor rots_0;
  torch::Tensor trans_1;
  torch::Tensor rots_1;
};
struct rot_t{
    float roll;
    float pitch;
    float yaw;
};
struct trans_t{
    float x;
    float y;
    float z;
};
struct transform_t{
    rot_t rot;
    trans_t trans;
};

struct vehicle_info_t{
    transform_t ego_motion;
    transform_t cam_extrinsic;
};

struct topic_data_t{
  topic_shared_ptr topic_ptr;
  u32 topic_size = 0;
};

struct topic_t{
  char service_name[128];
  char topic_name[128];
  u64 publish_time;
  u64 elapsed_time = 0;
  u64 frame_count = 0;
  topic_data_t topic_data;  
  void set_publish_times(u64 start_time = 0){
    publish_time = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
    elapsed_time = publish_time - start_time;
  }
};
