
// Standard includes
#include <iostream>
#include <sstream>
// System includes

// RT
#include <emulator/x330_umd/x330_regmap.h>

#include <emulator/x330_umd/x330_umd.hpp>
#include <unistd.h>

namespace sapeon::emul {

sapeon_result_t X330Umd::RunInference(
    const sapeon_size_t core_id, const sapeon_addr_t cps_addr,
    const sapeon_addr_t cps_size,
    const std::vector<sapeon_addr_t> &dma_regions) {
  // std::cout << "Run Inference on Core " << core_id << "at address 0x"
  //           << std::hex << cps_addr << " with size " << std::dec << cps_size
  //           << " and dma regions: {" << std::endl;
  // for (auto dma_region : dma_regions) {
  //   std::cout << "0x" << std::hex << dma_region << std::dec << std::endl;
  // }
  // std::cout << "}" << std::endl;
  std::unique_lock<std::mutex> lock(m_aicore_[core_id]);
  if (dma_regions.size() != kNumOfDmaRegions) {
    return SAPEON_INVALID_DMA_REGION_SIZE;
  }

  const sapeon_addr_t kBaseRegAddr =
      CORE_ADDRESS_SECTION_BASE + core_id * CORE_ADDRESS_SIZE;
  const sapeon_addr_t kGblCfgRegAddr = kBaseRegAddr + DCS_REG_GLOBAL_DESC_CFG;
  DcsRegGlobalDescConfig gbl_cfg{};
  RegRead(&gbl_cfg, kGblCfgRegAddr, sizeof(gbl_cfg));
  gbl_cfg.finish_intr_enable = 0;
  gbl_cfg.failure_intr_enable = 0;
  if (RegWrite(kGblCfgRegAddr, &gbl_cfg, sizeof(gbl_cfg)) != SAPEON_OK) {
    return SAPEON_REG_WRITE_FAILURE_GLOBAL_DESC_CFG;
  }

  DcsRegJobDesc cps_desc{};
  sapeon_addr_t encoded_cps_addr = EncodeDdrAddress(cps_addr);
  if (DecodeDdrAddress(encoded_cps_addr) != cps_addr) {
    return SAPEON_CPS_ADDR_NOT_ALIGNED;
  }
  cps_desc.cps_address = EncodeDdrAddress(cps_addr);

  if (cps_size % sizeof(u64) != 0) {
    return SAPEON_CPS_SIZE_NOT_ALIGNED;
  }
  cps_desc.cps_length = cps_size / sizeof(uint64_t);
  for (size_t i = 0; i < kNumOfDmaRegions; ++i) {
    sapeon_addr_t encoded_dma_region = EncodeDdrAddress(dma_regions[i]);
    if (DecodeDdrAddress(encoded_dma_region) != dma_regions[i]) {
      return SAPEON_DMA_REGION_NOT_ALIGNED;
    }
    cps_desc.dma_regions[i] = EncodeDdrAddress(dma_regions[i]);
  }
  cps_desc.hook_table = 0;
  cps_desc.cfg.valid = 1;

  const sapeon_addr_t kNextJobDescRegAddr =
      kBaseRegAddr + DCS_REG_NEXT_JOB_DESC;
  if (RegWrite(kNextJobDescRegAddr, &cps_desc, sizeof(cps_desc)) != SAPEON_OK) {
    return SAPEON_REG_WRITE_FAILURE_NEXT_JOB_DESC;
  }

  const sapeon_addr_t kDcsStatusAcRegAddr = kBaseRegAddr + DCS_REG_STATUS_AC;

  while (true) {
    DcsRegStatus status_ac{};
    if (RegRead(&status_ac, kDcsStatusAcRegAddr, sizeof(status_ac)) !=
        SAPEON_OK) {
      return SAPEON_REG_READ_FAILURE_STATUS_AC;
    }
    if (status_ac.checksum_error != 0) {
      return SAPEON_INFERENCE_ERROR_BAD_CHECKSUM;
    }
    if (status_ac.cps_rd_error != 0) {
      return SAPEON_INFERENCE_ERROR_READ_ERROR;
    }
    if (!status_ac.busy) break;
  }

  return SAPEON_OK;
}

sapeon_result_t X330Umd::SetGid(const sapeon_size_t core_id,
                                const sapeon_size_t gid) {
  return SAPEON_SET_GID_FAILURE;
}

sapeon_result_t X330Umd::RunIcvt(const sapeon_size_t icvt_id,
                                 const ICvtCommand &cmd) {
  // std::cout << "Run ICVT " << icvt_id << " on icvt " << icvt_id << std::endl;
  std::unique_lock<std::mutex> lock(m_icvt_[icvt_id]);
  sapeon_addr_t addr = get_lfcvt_reg_addr(icvt_id, LFCVT_REG_ICVT_NEXT_CMD);
  RegWrite(addr, &cmd, GetIcvtCmdSize(&cmd));
  addr = get_lfcvt_reg_addr(icvt_id, LFCVT_REG_ICVT_PUSH_CMD);
  u32 value = 1;
  RegWrite(addr, &value, sizeof(value));

  addr = get_lfcvt_reg_addr(icvt_id, LFCVT_REG_ICVT_STATUS_AC);
  while (true) {
    LfcvtRegStatus status;
    RegRead(&status, addr, sizeof(LfcvtRegStatus));
    // Todo : Need to find a way for checking success or fail in actual device.
    if (status.prev_finish) {
      return SAPEON_OK;
    }
  }
}

sapeon_result_t X330Umd::RunOcvt(const sapeon_size_t ocvt_id,
                                 const OCvtCommand &cmd) {
  // std::cout << "Run OCVT " << ocvt_id << " on ocvt " << ocvt_id << std::endl;
  std::unique_lock<std::mutex> lock(m_ocvt_[ocvt_id]);
  uint64_t addr = get_lfcvt_reg_addr(ocvt_id, LFCVT_REG_OCVT_NEXT_CMD);
  RegWrite(addr, &cmd, GetOcvtCmdSize());
  addr = get_lfcvt_reg_addr(ocvt_id, LFCVT_REG_OCVT_PUSH_CMD);
  u32 value = 1;
  RegWrite(addr, &value, sizeof(value));

  addr = get_lfcvt_reg_addr(ocvt_id, LFCVT_REG_OCVT_STATUS_AC);
  while (true) {
    LfcvtRegStatus status;
    RegRead(&status, addr, sizeof(LfcvtRegStatus));
    // Todo : Need to find a way for checking success or fail in actual device.
    if (status.prev_finish) {
      return SAPEON_OK;
    }
  }
}

sapeon_addr_t X330Umd::get_lfcvt_reg_addr(const sapeon_size_t icvt_id,
                                          const sapeon_size_t offset) const {
  sapeon_addr_t addr = offset;
  const u8 kLfcvtId = icvt_id / LFCVT_SUBCH_COUNT;
  const u8 kSubChId = icvt_id % LFCVT_SUBCH_COUNT;
  addr += LFCVT_ADDRESS_SECTION_BASE;
  addr += kLfcvtId * LFCVT_ADDRESS_SIZE;
  addr += kSubChId * LFCVT_SUBCH_ADDRESS_SIZE;
  return addr;
}

}  // namespace sapeon::emul