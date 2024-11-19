#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

void bev_pool_v2_kernel(int c, int n_intervals, const float *depth,
                        const float *feat, const int *ranks_depth,
                        const int *ranks_feat, const int *ranks_bev,
                        const int *interval_starts, const int *interval_lengths,
                        float *out) {
  for (int index = 0; index < n_intervals; ++index) {
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    for (int cur_c = 0; cur_c < c; ++cur_c) {
      float psum = 0;
      for (int i = 0; i < interval_length; ++i) {
        const float *cur_depth = depth + ranks_depth[interval_start + i];
        const float *cur_feat =
            feat + ranks_feat[interval_start + i] * c + cur_c;
        psum += *cur_feat * *cur_depth;
      }

      const int *cur_rank = ranks_bev + interval_start;
      float *cur_out = out + *cur_rank * c + cur_c;
      *cur_out = psum;
    }
  }
}

// Wrapper function to call the kernel
void bev_pool_v2_forward(const std::vector<float> &_depth,
                         const std::pair<std::vector<float>, int> &_feat,
                         std::vector<float> &_out,
                         const std::vector<int> &_ranks_depth,
                         const std::vector<int> &_ranks_feat,
                         const std::vector<int> &_ranks_bev,
                         const std::vector<int> &_interval_lengths,
                         const std::vector<int> &_interval_starts) {
  int c = _feat.second;
  int n_intervals = _interval_lengths.size();

  // Assuming data is already in the correct format and accessible
  const float *depth = _depth.data();
  const float *feat = _feat.first.data();
  const int *ranks_depth = _ranks_depth.data();
  const int *ranks_feat = _ranks_feat.data();
  const int *ranks_bev = _ranks_bev.data();

  const int *interval_lengths = _interval_lengths.data();
  const int *interval_starts = _interval_starts.data();

  float *out = _out.data();

  bev_pool_v2_kernel(c, n_intervals, depth, feat, ranks_depth, ranks_feat,
                     ranks_bev, interval_starts, interval_lengths, out);
}

template <typename T>
std::vector<T> readBinaryFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return {};
  }
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> fileData(fileSize / sizeof(T));
  file.read(reinterpret_cast<char *>(fileData.data()), fileSize);
  file.close();
  return fileData;
}

int main() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;

  if (cpuinfo.is_open()) {
    while (getline(cpuinfo, line)) {
      // Check for specific lines to print
      if (line.substr(0, 10) == "model name" ||
          line.substr(0, 7) == "cpu MHz" || line.substr(0, 9) == "processor") {
        std::cout << line << std::endl;
      }
      // To print the architecture, we look for the "flags" or "Features" line,
      // as the direct architecture info may not be clearly defined
      if ((line.substr(0, 5) == "flags" || line.substr(0, 8) == "Features")) {
        std::cout << "Architecture: ";
        // Simplified check: if it contains "lm" flag, it's likely x86_64;
        // otherwise, it's x86 (this is a simplification)
        if (line.find("lm") != std::string::npos) {
          std::cout << "x86_64" << std::endl;
        } else {
          std::cout << "x86" << std::endl;
        }
      }
    }
    cpuinfo.close();
  } else {
    std::cerr << "Unable to open /proc/cpuinfo" << std::endl;
    return 1;
  }
  std::cout << "PID of this program is: " << getpid() << std::endl;

  // clang-format off
  std::string data_root = "/bevdet/src/cpp/bev_pool_v2";
  auto depth = readBinaryFile<float>(data_root + "/depth_1_6_118_16_44_float32.bin");
  auto feat = readBinaryFile<float>(data_root + "/feat_1_6_16_44_80_float32.bin");
  auto ranks_depth = readBinaryFile<int>(data_root + "/ranks_depth_356967_int32.bin");
  auto ranks_feat = readBinaryFile<int>(data_root + "/ranks_feat_356967_int32.bin");
  auto ranks_bev = readBinaryFile<int>(data_root + "/ranks_bev_356967_int32.bin");
  auto interval_starts = readBinaryFile<int>(data_root + "/interval_starts_13407_int32.bin");
  auto interval_lengths = readBinaryFile<int>(data_root + "/interval_lengths_13407_int32.bin");
  auto bev_feat = readBinaryFile<float>(data_root + "/bev_feat_1_1_128_128_80_float32.bin");
  auto out = std::vector<float>(bev_feat.size(), 0);
  // clang-format on

  size_t iter = 100;
  std::cout << "Running the test " << iter << " times... This may take a while."
            << std::endl;

  double total_time = 0;
  for (size_t i = 0; i < iter; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    bev_pool_v2_forward(depth, {feat, 80}, out, ranks_depth, ranks_feat,
                        ranks_bev, interval_lengths, interval_starts);
    auto end = std::chrono::high_resolution_clock::now();

    total_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
  }

  std::cout << "Average latency: " << total_time / iter << " us" << std::endl;

  // Compare the output with the expected output
  for (size_t i = 0; i < out.size(); ++i) {
    if (std::abs(out[i] - bev_feat[i]) > 1e-6) {
      std::cerr << "Test failed at index " << i << " with value " << out[i]
                << " and expected " << bev_feat[i] << std::endl;
      return 1;
    }
  }

  std::cout << "Test passed." << std::endl;
  return 0;
}