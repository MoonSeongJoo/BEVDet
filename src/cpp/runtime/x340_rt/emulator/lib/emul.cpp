// Standard Library

// Project

#include <emulator/lib/emul.hpp>

#include "types.h"
#include "emul.hpp"
#include <iostream>

namespace sapeon::emul {

X340toX330Emul::X340toX330Emul(const std::string& server_address):x330_client_(std::make_unique<sapeon::emul::X330UmdAsic>(0))
    {
      device_status_ = x330_client_->Open();
    }

X340toX330Emul::X340toX330Emul(sapeon_size_t device_id):x330_client_(std::make_unique<sapeon::emul::X330UmdAsic>(device_id)){
   device_status_ = x330_client_->Open();
  } 

sapeon_result_t X340toX330Emul::DmaWrite(
    const sapeon_addr_t dst,
    const sapeon_byte_t *src,
    sapeon_size_t size)
{
    auto dst_ = static_cast<sapeon_addr_t>(dst);
    auto result = x330_client_->DmaWrite(
        dst, reinterpret_cast<const sapeon_byte_t *>(src),
        size);
    return result;
}
sapeon_result_t X340toX330Emul::DmaRead(sapeon_byte_t* dst,
                                        const sapeon_addr_t src,
                                        sapeon_size_t size) {
  auto src_ = static_cast<sapeon_addr_t>(src);
  auto result = x330_client_->DmaRead(dst, src_, size);                                          
  return result;
}

sapeon_result_t X340toX330Emul::RunIcvt(const sapeon_size_t icvt_id,
                                        const ICvtCommand& cmd) {
  return x330_client_->RunIcvt(icvt_id, cmd);
}

sapeon_result_t X340toX330Emul::RunOcvt(const sapeon_size_t ocvt_id,
                                        const OCvtCommand& cmd) {
  return x330_client_->RunOcvt(ocvt_id, cmd);
}

sapeon_result_t X340toX330Emul::RunInference(
    const sapeon_size_t core_id, const sapeon_addr_t cps_addr,
    const sapeon_addr_t cps_size,
    const std::vector<sapeon_addr_t>& dma_regions) {
  return x330_client_->RunInference(core_id, cps_addr, cps_size, dma_regions);
}

sapeon_result_t X340toX330Emul::SetGid(const sapeon_size_t core_id,
                                       const sapeon_size_t gid) {
  return x330_client_->SetGid(core_id, gid);
}

sapeon_result_t X340toX330Emul::GetDeviceStatus(){
  return device_status_;
}
}  // namespace sapeon::emul
