#pragma once
#include "Topics.hpp"
#include <string>
#include <types.h>
#include <chrono>
#include <string>
#include <exception>
#include <thread>
#include <iostream>

struct dma_file_info_t {
  std::string filename;
  sapeon_addr_t addr;
  sapeon_size_t size;
};

struct inference_info_t {
  int32_t core_id;
  int32_t group_id;
  u64 cps_addr;
  u64 cps_size;
  sapeon_addr_t dma_regions[4];
};

struct icvt_info_t {
  ICvtCommand icvt_cmd;
  uint32_t core_id;
};

struct ocvt_info_t {
  OCvtCommand ocvt_cmd;
  uint32_t core_id;
};

