#pragma once

// Standard Library
#include <memory>
#include <vector>
// Project

#include <emulator/x330_umd/x330_regmap.h>
#include <emulator/x330_umd/x330_umd_asic.hpp>

#include "types.h"

namespace sapeon::emul {
class X340toX330Emul {
 public:
  X340toX330Emul(const std::string& server_address);
  X340toX330Emul(sapeon_size_t device_id);
  sapeon_result_t DmaWrite(const sapeon_addr_t dst, const sapeon_byte_t* src,
                           sapeon_size_t size);
  sapeon_result_t DmaRead(sapeon_byte_t* dst,
                          const sapeon_addr_t src, sapeon_size_t size);

  sapeon_result_t RunIcvt(const sapeon_size_t icvt_id, const ICvtCommand& cmd);
  sapeon_result_t RunOcvt(const sapeon_size_t ocvt_id, const OCvtCommand& cmd);
  sapeon_result_t RunInference(const sapeon_size_t core_id,
                               const sapeon_addr_t cps_addr,
                               const sapeon_addr_t cps_size,
                               const std::vector<sapeon_addr_t>& dma_regions);
  sapeon_result_t SetGid(const sapeon_size_t core_id, const sapeon_size_t gid);

  sapeon_result_t GetDeviceStatus();

  private:
  std::vector<bool> core_status_;
  std::vector<bool> icvt_status_;
  std::vector<bool> ocvt_status_;
  std::unique_ptr<sapeon::emul::X330UmdAsic> x330_client_;
  sapeon_result_t device_status_;
};
}  // namespace sapeon::emul
