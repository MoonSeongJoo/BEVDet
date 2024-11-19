#pragma once

// Standard includes
#include <mutex>

// System includes
#include <semaphore.h>

// RT
#include <emulator/x330_umd/x330_umd.hpp>

namespace sapeon::emul {
class X330UmdAsic : public X330Umd {
 public:
  X330UmdAsic(sapeon_size_t device_id) : X330Umd(device_id) {}
  virtual ~X330UmdAsic() = default;

  virtual sapeon_result_t Open() override;
  virtual sapeon_result_t Close() override;
  virtual sapeon_result_t DmaWrite(const sapeon_addr_t dst, const sapeon_byte_t *src,
                                    sapeon_size_t size) override;
  virtual sapeon_result_t DmaRead(sapeon_byte_t *dst, const sapeon_addr_t src,
                                   sapeon_size_t size) override;

 protected:
  virtual sapeon_result_t RegWrite(const sapeon_addr_t addr, const void *src,
                                   const sapeon_size_t size) override;
  virtual sapeon_result_t RegRead(void *dst, const sapeon_addr_t addr,
                                  const sapeon_size_t size) override;

 private:
  int user_fd_ = -1;
  volatile void *user_ptr_ = nullptr;

  const sapeon_size_t kMmapSize = 4096 * 16;
  static constexpr sapeon_size_t kNumOfDMAChannels = 8;

  u32 setHiAddrAndGetLoAddr(u32 addr);

  std::mutex m_chip_;
  std::mutex m_aicore_[kNumOfCores];
  std::mutex m_icvt_[kNumOfCores];
  std::mutex m_ocvt_[kNumOfCores];
  sem_t s_dma_write_;
  sem_t s_dma_read_;
  std::mutex m_reg_;
};
}  // namespace sapeon::emul