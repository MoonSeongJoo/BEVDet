
// Standard includes
#include <sstream>

// System includes
#include <fcntl.h>
#include <semaphore.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <iostream>

// RT
#include <emulator/x330_umd/x330_umd_asic.hpp>

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

namespace sapeon::emul {

static inline u64 IDivCeil(u64 x, u64 y) { return (x + y - 1) / y; }

sapeon_result_t X330UmdAsic::Open() {
  std::stringstream ss;
  ss << "/dev/snx3-" << device_id();
  user_fd_ = open(ss.str().c_str(), O_RDWR);
  if (user_fd_ < 0) {
    return SAPEON_DEVICE_OPEN_FAILURE;
  }
  user_ptr_ =
      mmap(0, kMmapSize, PROT_READ | PROT_WRITE, MAP_SHARED, user_fd_, 0);
  if (user_ptr_ == MAP_FAILED) {
    return SAPEON_DEVICE_MMAP_FAILURE;
  }

  sem_init(&s_dma_write_, 0, kNumOfDMAChannels);
  sem_init(&s_dma_read_, 0, kNumOfDMAChannels);

  for (sapeon_size_t core_id = 0; core_id < kNumOfCores; ++core_id) {

    const sapeon_addr_t kBaseRegAddr =
        CORE_ADDRESS_SECTION_BASE + core_id * CORE_ADDRESS_SIZE;
    const sapeon_addr_t kDcsStatusAcRegAddr = kBaseRegAddr + DCS_REG_STATUS_AC;

    u32 reg_addr = (kDcsStatusAcRegAddr & 0xffffffff);
    setHiAddrAndGetLoAddr(reg_addr);

    DcsRegStatus status_ac{};
    if (RegRead(&status_ac, kDcsStatusAcRegAddr, sizeof(status_ac)) !=
        SAPEON_OK) {
      return SAPEON_REG_READ_FAILURE_STATUS_AC;
    }
    status_ac.raw_u32 = 0;
    if (RegWrite(kDcsStatusAcRegAddr, &status_ac, sizeof(status_ac)) !=
        SAPEON_OK) {
      return SAPEON_REG_WRITE_FAILURE_STATUS_AC;
    }

    float g_ga_up_steps_usec = 16.0;
    float g_ga_dn_steps_usec = 16.0;
    int g_ga_min_active_level = 16;
    int g_ga_up_tracking_only = 0;
    int g_ga_dn_tracking_only = 0;
    int g_ga_dn_steps_on_alert = 0;
    int g_ga_alert_det_cycles = 0;
    int g_ga_backp_det_cycles = 32;

    // Apply the Gradual Activation
    // It remove unstable result in full clock speed.
    struct GradActParam {
      float core_clk_mhz;
      float up_steps_usec;
      float dn_steps_usec;
      int min_active_level;
      bool up_tracking_only;
      bool dn_tracking_only;
      int dn_steps_on_alert;
      int alert_det_cycles;
      int backp_det_cycles;
    };
    GradActParam param = {};
    param.core_clk_mhz = 700;
    param.up_steps_usec = g_ga_up_steps_usec;
    param.dn_steps_usec = g_ga_dn_steps_usec;
    param.min_active_level = g_ga_min_active_level;
    param.up_tracking_only = g_ga_up_tracking_only;
    param.dn_tracking_only = g_ga_dn_tracking_only;
    param.dn_steps_on_alert = g_ga_dn_steps_on_alert;
    param.alert_det_cycles = g_ga_alert_det_cycles;
    param.backp_det_cycles = g_ga_backp_det_cycles;

    int NSTEPS = 16;
    int up_step_cycles =
        (param.up_steps_usec * param.core_clk_mhz / (NSTEPS - 1));
    int dn_step_cycles =
        (param.dn_steps_usec * param.core_clk_mhz / (NSTEPS - 1));

    DcsRegGlobalDescMxcGradAct mxc_ga = {};
    mxc_ga.up_tracking_only = param.up_tracking_only ? 1 : 0;
    mxc_ga.dn_tracking_only = param.dn_tracking_only ? 1 : 0;
    mxc_ga.dn_steps_on_alert = param.dn_steps_on_alert;
    mxc_ga.min_active_level = param.min_active_level;
    mxc_ga.alert_det_cycles = param.alert_det_cycles;
    mxc_ga.backp_det_cycles = param.backp_det_cycles;
    mxc_ga.up_step_cycles = up_step_cycles;
    mxc_ga.dn_step_cycles = dn_step_cycles;

    u64 mxc_grad_act_reg = kBaseRegAddr + DCS_REG_GLOBAL_DESC_MXC_GRAD_ACT;
    if (RegWrite(mxc_grad_act_reg, &mxc_ga, sizeof(mxc_ga)) != SAPEON_OK) {
      return SAPEON_REG_WRITE_FAILURE_GLOBAL_DESC_MXC_GRAD_ACT;
    }
  }

  return SAPEON_OK;
}

sapeon_result_t X330UmdAsic::Close() {
  if (user_ptr_ != nullptr) {
    munmap(const_cast<void *>(user_ptr_), kMmapSize);
    user_ptr_ = nullptr;
  }
  if (user_fd_ != 0) {
    close(user_fd_);
  }

  sem_destroy(&s_dma_write_);
  sem_destroy(&s_dma_read_);

  return SAPEON_OK;
}

sapeon_result_t X330UmdAsic::DmaWrite(const sapeon_addr_t dst,
                                       const sapeon_byte_t *src,
                                       const sapeon_size_t size) {
  // std::cout << "DramWrite: dst=" << std::hex << dst << ", size=" << std::dec
  //           << size << std::endl;
  sem_wait(&s_dma_write_);
  const sapeon_size_t kMaxBlkSize = 1 << 24;
  const u64 iter = IDivCeil(size, kMaxBlkSize);
  for (u64 i = 0; i < iter; ++i) {
    sapeon_size_t pos = i * kMaxBlkSize;
    sapeon_size_t remained = size - pos;
    sapeon_size_t rsize = std::min(kMaxBlkSize, remained);
    u64 raddr = dst + pos;
    u64 rbuf = (u64)(src) + pos;
    sapeon_size_t rc;
    rc = pwrite(user_fd_, reinterpret_cast<void *>(rbuf), rsize, raddr);
    if (rc != rsize) {
      sem_post(&s_dma_write_);
      return SAPEON_DMA_H2D_FAILURE;
    }
  }
  sem_post(&s_dma_write_);
  return SAPEON_OK;
}

sapeon_result_t X330UmdAsic::DmaRead(sapeon_byte_t *dst,
                                      const sapeon_addr_t src,
                                      const sapeon_size_t size) {
  // std::cout << "DramRead: src=" << std::hex << src << ", size=" << std::dec
  //           << size << std::endl;

  u64 tic, toc;
  u64 sum = 0;

  sem_wait(&s_dma_read_);
  const size_t kMaxBlkSize = 1 << 24;
  const u64 iter = IDivCeil(size, kMaxBlkSize);

  for (u64 i = 0; i < iter; ++i) {
    sapeon_size_t pos = i * kMaxBlkSize;
    sapeon_size_t remained = size - pos;
    sapeon_size_t rsize = std::min(kMaxBlkSize, remained);
    u64 raddr = src + pos;
    u64 rbuf = (u64)dst + pos;
    sapeon_size_t rc;
    rc = pread(user_fd_, reinterpret_cast<void *>(rbuf), rsize, raddr);
    if (rc != rsize) {
      sem_post(&s_dma_read_);
      return SAPEON_DMA_D2H_FAILURE;
    }
  }
  sem_post(&s_dma_read_);
  // std::cout << "DramRead Data[0] : 0x" << std::hex << (int)(u8)(dst[0])
  //           << std::endl;
  return SAPEON_OK;
}

sapeon_result_t X330UmdAsic::RegWrite(const sapeon_addr_t addr, const void *src,
                                      const sapeon_size_t size) {
  std::unique_lock<std::mutex> lock(m_reg_);
  const u32 kLoAddrOffset = 0xC000;
  u32 lo_addr = (addr & 0x3FFF) | kLoAddrOffset;
  u32 word_size = size / sizeof(u32);
  const u32 *word_ptr = reinterpret_cast<const u32 *>(src);
  for (u32 i = 0; i < word_size; i++) {
    *(volatile u32 *)((u64)user_ptr_ + lo_addr + sizeof(u32) * i) = *word_ptr++;
  }
  return SAPEON_OK;
}

sapeon_result_t X330UmdAsic::RegRead(void *dst, const sapeon_addr_t addr,
                                     const sapeon_size_t size) {
  std::unique_lock<std::mutex> lock(m_reg_);
  const uint32_t kLoAddrOffset = 0xC000;
  uint32_t lo_addr = (addr & 0x3FFF) | kLoAddrOffset;

  uint32_t word_size = size / sizeof(uint32_t);
  uint32_t *word_ptr = reinterpret_cast<uint32_t *>(dst);
  for (uint32_t i = 0; i < word_size; i++) {
    *word_ptr = *reinterpret_cast<volatile uint32_t *>(
        reinterpret_cast<uint64_t>(user_ptr_) + lo_addr + sizeof(uint32_t) * i);
    word_ptr++;
  }
  return SAPEON_OK;
}

u32 X330UmdAsic::setHiAddrAndGetLoAddr(u32 addr) {
  const u32 kHiAddrSetAddr = 0x8000;  // 32kB
  const u32 kLoAddrOffset = 0xC000;

  static u32 curr_high_addr;

  u32 high_addr = addr >> 14;
  if (high_addr < 0x4) {
    return addr;
  } else if (curr_high_addr != high_addr) {
    curr_high_addr = high_addr;

    u32 commbox_setting = high_addr << 10;
    *reinterpret_cast<u32 *>(reinterpret_cast<uint64_t>(user_ptr_) +
                             kHiAddrSetAddr) = commbox_setting;
  }
  u32 lo_addr = (addr & 0x3FFF) | kLoAddrOffset;
  return lo_addr;
}

}  // namespace sapeon::emul


