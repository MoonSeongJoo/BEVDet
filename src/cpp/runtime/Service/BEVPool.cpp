#include "BEVPool.hpp" // Includes the header file for the BEVPool class

// Defines the Run method for the BEVPool class
sapeon_result_t BEVPool::Run(topic_t subscribed_topic)
{
    // Retrieves synchronized topics based on the subscribed topic and the subscription list
    auto sync_topics = service_sync_.GetSync(subscribed_topic, subscribe_list_);
    // Checks if there are any synchronized topics available
    if (!sync_topics.empty())
    {
        // Attempts to retrieve pre-allocated memory for the BEV feature data
        auto topic_data = GetData("bev_feature_pre_single");
        // If no pre-allocated memory is available, allocate new memory
        if(topic_data.topic_ptr == nullptr){
            // Calculates the size needed for the topic data based on feature numbers and feature map dimensions
            topic_data.topic_size = feature_nums_ * feature_map_shape_[0] * feature_map_shape_[1] * sizeof(float);
            // Allocates memory for the topic data
            topic_data.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_data.topic_size]);
        }
        // Extracts the feature and depth topics from the synchronized topics
        auto feature_topic = sync_topics["feature"];
        auto depth_topic = sync_topics["depth"];
        // Interprets the raw byte data as float arrays for feature and depth data
        auto feature_data = reinterpret_cast<const float*>(feature_topic.topic_data.topic_ptr.get());
        auto depth_data = reinterpret_cast<const float*>(depth_topic.topic_data.topic_ptr.get());
        // Interprets the allocated memory for output data as a float array
        auto out = reinterpret_cast<float*>(topic_data.topic_ptr.get());
        memset(reinterpret_cast<void*>(out), 0, topic_data.topic_size);
        // Calls the bevPoolV2Forward function to process the depth and feature data
        bevPoolV2Forward(depth_data, {feature_data, feature_nums_}, out, ranks_depth, ranks_feat,ranks_bev, interval_lengths, interval_starts);
        // Publishes the processed BEV feature data
        Publish(subscribed_topic.frame_count, "bev_feature_pre_single", topic_data);
    }
    // Returns SAPEON_OK indicating successful execution
    return SAPEON_OK;
}

// Defines the bevPoolV2Kernel function, which performs the BEV pooling operation
void BEVPool::bevPoolV2Kernel(int c, int n_intervals, const float *depth,
                                 const float *feat, const size_t *ranks_depth,
                                 const size_t *ranks_feat, const size_t *ranks_bev,
                                 const size_t *interval_starts, const size_t *interval_lengths,
                                 float *out)
{
    // Iterates over each interval
    for (int index = 0; index < n_intervals; ++index)
    {
        // Retrieves the start index and length of the current interval
        int interval_start = interval_starts[index];
        int interval_length = interval_lengths[index];
        // Iterates over each channel
        for (int cur_c = 0; cur_c < c; ++cur_c)
        {
            // Initializes the partial sum for the current channel to 0
            float psum = 0;
            // Iterates over each element in the interval
            for (int i = 0; i < interval_length; ++i)
            {
                // Calculates the current depth and feature pointers based on ranks
                const float *cur_depth = depth + ranks_depth[interval_start + i];
                const float *cur_feat = feat + ranks_feat[interval_start + i] * c + cur_c;
                // Accumulates the product of the current feature and depth values
                psum += *cur_feat * *cur_depth;
            }

            // Calculates the output pointer based on the current rank and channel
            const size_t *cur_rank = ranks_bev + interval_start;
            float *cur_out = out + *cur_rank * c + cur_c;
            // Stores the accumulated sum in the output array
            *cur_out = psum;
        }
    }
}

// Improved kernel function for BEV pooling operation
void BEVPool::BevPoolV2KernelImproved(int c, int n_intervals, const float *depth,
                        const float *feat, const size_t *ranks_depth,
                        const size_t *ranks_feat, const size_t *ranks_bev,
                        const size_t *interval_starts, const size_t *interval_lengths,
                        float *out) {
  // Iterates over each interval
  for (int index = 0; index < n_intervals; ++index) {
    // Determines the start of the current interval
    int interval_start = interval_starts[index];
    // Determines the length of the current interval
    int interval_length = interval_lengths[index];
    // Calculates the output index based on the rank of the current interval
    const int cur_rank = ranks_bev[interval_start];
    // Points to the correct location in the output buffer to store results
    float *cur_out = &out[cur_rank * c];

    // Iterates over each element within the current interval
    for (int i = 0; i < interval_length; ++i) {
        // Retrieves the depth value for the current element
        const float cur_depth = depth[ranks_depth[interval_start + i]];
        // Points to the feature data for the current element
        const float *cur_feat = &feat[ranks_feat[interval_start + i] * c];
        // Accumulates weighted feature values into the output buffer
        for (int cur_c = 0; cur_c < c; ++cur_c) {
          cur_out[cur_c] += cur_feat[cur_c] * cur_depth;
        }
      }
  }
}

// Wrapper function to prepare and call the improved BEV pooling kernel
void BEVPool::bevPoolV2Forward(const float* _depth,
                         const std::pair<const float*, int> _feat,
                         float* _out,
                         const std::vector<size_t> &_ranks_depth,
                         const std::vector<size_t> &_ranks_feat,
                         const std::vector<size_t> &_ranks_bev,
                         const std::vector<size_t> &_interval_lengths,
                         const std::vector<size_t> &_interval_starts) {
  // Extracts the number of channels from the feature data
  int c = _feat.second;
  // Determines the number of intervals to process
  int n_intervals = _interval_lengths.size();

  // Sets up pointers and sizes for depth data, feature data, and various ranks and intervals
  const float *depth = _depth;
  const float *feat = _feat.first;
  const size_t *ranks_depth = _ranks_depth.data();
  const size_t *ranks_feat = _ranks_feat.data();
  const size_t *ranks_bev = _ranks_bev.data();
  const size_t *interval_lengths = _interval_lengths.data();
  const size_t *interval_starts = _interval_starts.data();

  // Points to the output buffer where results will be stored
  float *out = _out;
  
  // Calls the improved kernel function with prepared data
  BevPoolV2KernelImproved(c, n_intervals, depth, feat, ranks_depth, ranks_feat,
                     ranks_bev, interval_starts, interval_lengths, out);
}

std::vector<float> BEVPool::getLidarCoor(
    size_t B, size_t N, const std::vector<float> &frustum,
    const std::vector<size_t> &shape, const std::vector<float> &rots,
    const std::vector<float> &trans, const std::vector<float> &cam2imgs,
    const std::vector<float> &post_rots, const std::vector<float> &post_trans,
    const std::vector<float> &bda) {

  // frustum: D x H x W x 3 -> points: B x N x D x H x W x 3
  // points = points - post_trans
  Eigen::DenseIndex D = shape[0], H = shape[1], W = shape[2], C = shape[3];
  Eigen::Tensor<float, 6> points(B, N, D, H, W, C);
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      for (Eigen::DenseIndex d = 0; d < D; ++d) {
        for (Eigen::DenseIndex h = 0; h < H; ++h) {
          for (Eigen::DenseIndex w = 0; w < W; ++w) {
            for (Eigen::DenseIndex c = 0; c < C; ++c) {
              points(b, n, d, h, w, c) =
                  frustum[d * H * W * C + h * W * C + w * C + c] -
                  post_trans[n * 3 + c];
            }
          }
        }
      }
    }
  }

  // prepare post_rots_inv
  Eigen::Tensor<float, 4> post_rots_inv(B, N, 3, 3);
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      // inverse
      Eigen::MatrixXf mat(3, 3);
      for (Eigen::DenseIndex i = 0; i < 3; ++i) {
        for (Eigen::DenseIndex j = 0; j < 3; ++j) {
          mat(i, j) = post_rots[n * 9 + i * 3 + j];
        }
      }
      mat = mat.inverse();
      for (Eigen::DenseIndex i = 0; i < 3; ++i) {
        for (Eigen::DenseIndex j = 0; j < 3; ++j) {
          post_rots_inv(b, n, i, j) = mat(i, j);
        }
      }
    }
  }

  // points = post_rots_inv * points
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      for (Eigen::DenseIndex d = 0; d < D; ++d) {
        for (Eigen::DenseIndex h = 0; h < H; ++h) {
          for (Eigen::DenseIndex w = 0; w < W; ++w) {
            Eigen::Tensor<float, 1> point =
                points.chip<0>(b).chip<0>(n).chip<0>(d).chip<0>(h).chip<0>(w);
            Eigen::Tensor<float, 1> tmp_point = point;
            point = post_rots_inv.chip<0>(b).chip<0>(n).contract(
                tmp_point, Eigen::array<Eigen::IndexPair<int>, 1>{
                               Eigen::IndexPair<int>(1, 0)});
            // update
            points.chip<0>(b).chip<0>(n).chip<0>(d).chip<0>(h).chip<0>(w) =
                point;
          }
        }
      }
    }
  }

  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      for (Eigen::DenseIndex d = 0; d < D; ++d) {
        for (Eigen::DenseIndex h = 0; h < H; ++h) {
          for (Eigen::DenseIndex w = 0; w < W; ++w) {
            points(b, n, d, h, w, 0) =
                points(b, n, d, h, w, 0) * points(b, n, d, h, w, 2);
            points(b, n, d, h, w, 1) =
                points(b, n, d, h, w, 1) * points(b, n, d, h, w, 2);
          }
        }
      }
    }
  }

  Eigen::Tensor<float, 4> cam2ego(B, N, 3, 3);
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {

      Eigen::MatrixXf rot(3, 3);
      for (Eigen::DenseIndex i = 0; i < 3; ++i) {
        for (Eigen::DenseIndex j = 0; j < 3; ++j) {
          rot(i, j) = rots[b * N * 9 + n * 9 + i * 3 + j];
        }
      }

      Eigen::MatrixXf mat(3, 3);
      for (Eigen::DenseIndex i = 0; i < 3; ++i) {
        for (Eigen::DenseIndex j = 0; j < 3; ++j) {
          mat(i, j) = cam2imgs[b * N * 9 + n * 9 + i * 3 + j];
        }
      }

      mat = mat.inverse();

      mat = rot * mat;

      for (Eigen::DenseIndex i = 0; i < 3; ++i) {
        for (Eigen::DenseIndex j = 0; j < 3; ++j) {
          cam2ego(b, n, i, j) = mat(i, j);
        }
      }
    }
  }

  // points = combine * points + trans
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      for (Eigen::DenseIndex d = 0; d < D; ++d) {
        for (Eigen::DenseIndex h = 0; h < H; ++h) {
          for (Eigen::DenseIndex w = 0; w < W; ++w) {
            Eigen::MatrixXf combine(3, 3);
            for (Eigen::DenseIndex i = 0; i < 3; ++i) {
              for (Eigen::DenseIndex j = 0; j < 3; ++j) {
                combine(i, j) = cam2ego(b, n, i, j);
              }
            }

            Eigen::MatrixXf point(3, 1);
            point(0, 0) = points(b, n, d, h, w, 0);
            point(1, 0) = points(b, n, d, h, w, 1);
            point(2, 0) = points(b, n, d, h, w, 2);

            point = combine * point;

            point(0, 0) = point(0, 0) + trans[n * 3 + 0];
            point(1, 0) = point(1, 0) + trans[n * 3 + 1];
            point(2, 0) = point(2, 0) + trans[n * 3 + 2];

            Eigen::MatrixXf mat_bda(3, 3);
            for (Eigen::DenseIndex i = 0; i < 3; ++i) {
              for (Eigen::DenseIndex j = 0; j < 3; ++j) {
                mat_bda(i, j) = bda[b * 9 + i * 3 + j];
              }
            }

            point = mat_bda * point;

            points(b, n, d, h, w, 0) = point(0, 0);
            points(b, n, d, h, w, 1) = point(1, 0);
            points(b, n, d, h, w, 2) = point(2, 0);
          }
        }
      }
    }
  }

  std::vector<float> points_vec(B * N * D * H * W * 3);
  for (Eigen::DenseIndex b = 0; b < B; ++b) {
    for (Eigen::DenseIndex n = 0; n < N; ++n) {
      for (Eigen::DenseIndex d = 0; d < D; ++d) {
        for (Eigen::DenseIndex h = 0; h < H; ++h) {
          for (Eigen::DenseIndex w = 0; w < W; ++w) {
            for (Eigen::DenseIndex c = 0; c < 3; ++c) {
              points_vec[b * N * D * H * W * 3 + n * D * H * W * 3 +
                         d * H * W * 3 + h * W * 3 + w * 3 + c] =
                  points(b, n, d, h, w, c);
            }
          }
        }
      }
    }
  }
  // B x N x D x H x W x 3
  return points_vec;
}

std::pair<std::vector<float>, std::vector<size_t>> BEVPool::createFrustum(float lower_bound, float upper_bound, float interval,
               float input_height, float input_width, float downsample) {
  int H_feat = static_cast<int>(input_height / downsample);
  int W_feat = static_cast<int>(input_width / downsample);
  int D = static_cast<int>((upper_bound - lower_bound) / interval);

  std::vector<float> frustum(D * H_feat * W_feat * 3);
  std::vector<size_t> shape = {static_cast<size_t>(D),
                               static_cast<size_t>(H_feat),
                               static_cast<size_t>(W_feat), 3};

  for (int d = 0; d < D; ++d) {
    float depth_value = lower_bound + d * interval;
    for (int h = 0; h < H_feat; ++h) {
      float y = static_cast<float>(h) * (input_height - 1.0) / (H_feat - 1);
      for (int w = 0; w < W_feat; ++w) {
        float x = static_cast<float>(w) * (input_width - 1.0) / (W_feat - 1);

        // Calculate the index in the 1D vector
        int index = (d * H_feat * W_feat + h * W_feat + w) * 3;

        // Assign values to the 1D vector
        frustum[index] = x;               // X-coordinate
        frustum[index + 1] = y;           // Y-coordinate
        frustum[index + 2] = depth_value; // Depth value
      }
    }
  }

  // D x H x W x 3
  return {frustum, shape};
}

BEVPool::voxel_pool_output_t BEVPool::voxelPoolingPrepareV2(size_t B, size_t N, const std::vector<float> &coor,
                         const std::vector<size_t> &shape,
                         const std::vector<float> &grid_lower_bound,
                         const std::vector<float> &grid_interval,
                         const std::vector<int> &grid_size) {

  size_t D = shape[0];
  size_t H = shape[1];
  size_t W = shape[2];
  size_t num_points = B * N * D * H * W;

  std::vector<size_t> ranks_depth(num_points);
  std::iota(ranks_depth.begin(), ranks_depth.end(), 0);

  std::vector<size_t> ranks_feat(num_points / D);
  std::iota(ranks_feat.begin(), ranks_feat.end(), 0);

  std::vector<size_t> ranks_feat_reshaped(B * N * D * H * W);
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < D; ++k) {
        int idx1 = (i * N * D + j * D + k) * H * W;
        int idx2 = (i * N + j) * H * W;
        std::copy(ranks_feat.begin() + idx2, ranks_feat.begin() + idx2 + H * W,
                  ranks_feat_reshaped.begin() + idx1);
      }
    }
  }

  std::vector<std::vector<size_t>> coor_copy(num_points,
                                             std::vector<size_t>(4));
  for (int i = 0; i < B; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < D; ++k) {
        for (int l = 0; l < H; ++l) {
          for (int m = 0; m < W; ++m) {
            coor_copy[i * N * D * H * W + j * D * H * W + k * H * W + l * W + m]
                     [0] = (coor[i * N * D * H * W * 3 + j * D * H * W * 3 +
                                 k * H * W * 3 + l * W * 3 + m * 3 + 0] -
                            grid_lower_bound[0]) /
                           grid_interval[0];
            coor_copy[i * N * D * H * W + j * D * H * W + k * H * W + l * W + m]
                     [1] = (coor[i * N * D * H * W * 3 + j * D * H * W * 3 +
                                 k * H * W * 3 + l * W * 3 + m * 3 + 1] -
                            grid_lower_bound[1]) /
                           grid_interval[1];
            coor_copy[i * N * D * H * W + j * D * H * W + k * H * W + l * W + m]
                     [2] = (coor[i * N * D * H * W * 3 + j * D * H * W * 3 +
                                 k * H * W * 3 + l * W * 3 + m * 3 + 2] -
                            grid_lower_bound[2]) /
                           grid_interval[2];
            coor_copy[i * N * D * H * W + j * D * H * W + k * H * W + l * W + m]
                     [3] = i;
          }
        }
      }
    }
  }

  std::vector<size_t> kept;
  for (int i = 0; i < num_points; ++i) {
    if (coor_copy[i][0] >= 0 && coor_copy[i][0] < grid_size[0] &&
        coor_copy[i][1] >= 0 && coor_copy[i][1] < grid_size[1] &&
        coor_copy[i][2] >= 0 && coor_copy[i][2] < grid_size[2]) {
      kept.push_back(i);
    }
  }

  std::vector<size_t> coor_kept(kept.size() * 4), ranks_depth_kept(kept.size()),
      ranks_feat_kept(kept.size());

  for (int i = 0; i < kept.size(); ++i) {
    coor_kept[i * 4] = coor_copy[kept[i]][0];
    coor_kept[i * 4 + 1] = coor_copy[kept[i]][1];
    coor_kept[i * 4 + 2] = coor_copy[kept[i]][2];
    coor_kept[i * 4 + 3] = coor_copy[kept[i]][3];
    ranks_depth_kept[i] = ranks_depth[kept[i]];
    ranks_feat_kept[i] = ranks_feat_reshaped[kept[i]];
  }

  std::vector<size_t> ranks_bev(kept.size());
  for (int i = 0; i < kept.size(); ++i) {
    ranks_bev[i] =
        coor_kept[i * 4] + coor_kept[i * 4 + 1] * grid_size[0] +
        coor_kept[i * 4 + 2] * grid_size[0] * grid_size[1] +
        coor_kept[i * 4 + 3] * grid_size[0] * grid_size[1] * grid_size[2];
  }

  std::vector<size_t> order(kept.size());
  std::iota(order.begin(), order.end(), 0);

  std::sort(order.begin(), order.end(), [&ranks_bev](size_t i1, size_t i2) {
    return ranks_bev[i1] != ranks_bev[i2] ? ranks_bev[i1] < ranks_bev[i2]
                                          : i1 < i2;
  });

  std::vector<size_t> ranks_bev_sorted(kept.size());
  std::vector<size_t> ranks_depth_sorted(kept.size());
  std::vector<size_t> ranks_feat_sorted(kept.size());
  for (int i = 0; i < kept.size(); ++i) {
    ranks_bev_sorted[i] = ranks_bev[order[i]];
    ranks_depth_sorted[i] = ranks_depth_kept[order[i]];
    ranks_feat_sorted[i] = ranks_feat_kept[order[i]];
  }

  std::vector<size_t> interval_starts;
  interval_starts.push_back(0);
  for (size_t i = 1; i < ranks_bev_sorted.size(); ++i) {
    if (ranks_bev_sorted[i] != ranks_bev_sorted[i - 1]) {
      interval_starts.push_back(i);
    }
  }

  std::vector<size_t> interval_lengths;
  interval_lengths.resize(interval_starts.size());
  std::adjacent_difference(interval_starts.begin(), interval_starts.end(),
                           interval_lengths.begin());
  // rotate to the left 1 step
  std::rotate(interval_lengths.begin(), interval_lengths.begin() + 1,
              interval_lengths.end());
  interval_lengths.back() = ranks_bev_sorted.size() - interval_starts.back();

  return voxel_pool_output_t({ranks_bev_sorted, ranks_depth_sorted, ranks_feat_sorted, interval_starts, interval_lengths});
//   return std::make_tuple(ranks_bev_sorted, ranks_depth_sorted,
//                          ranks_feat_sorted, interval_starts, interval_lengths);
}