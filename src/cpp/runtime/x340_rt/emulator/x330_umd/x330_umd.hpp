#pragma once

// Standard includes
#include <mutex>
#include <vector>

// System includes

// RT
#include <emulator/x330_umd/x330_regmap.h>
#include <types.h>
#include <emulator/lib/x330.hpp>

namespace sapeon::emul
{
    class X330Umd : public X330Client
    {
    public:
        X330Umd(sapeon_size_t device_id) : kDeviceId_(device_id) {}
        virtual ~X330Umd() = default;

        virtual sapeon_result_t Open() = 0;
        virtual sapeon_result_t Close() = 0;
        virtual sapeon_result_t DmaWrite(const sapeon_addr_t dst,
                                          const sapeon_byte_t *src,
                                          const sapeon_size_t size) = 0;
                                          
        virtual sapeon_result_t DmaRead(sapeon_byte_t *dst,
                                        const sapeon_addr_t src,
                                        sapeon_size_t size) = 0;

        virtual sapeon_result_t RunInference(
            const sapeon_size_t core_id, const sapeon_addr_t cps_addr,
            const sapeon_addr_t cps_size,
            const std::vector<sapeon_addr_t> &dma_regions);
        virtual sapeon_result_t RunIcvt(const sapeon_size_t icvt_id,
                                        const ICvtCommand &cmd);
        virtual sapeon_result_t RunOcvt(const sapeon_size_t ocvt_id,
                                        const OCvtCommand &cmd);
        virtual sapeon_result_t SetGid(const sapeon_size_t core_id,
                                       const sapeon_size_t gid);
        sapeon_size_t GetNumCore()
        {
            return kNumOfCores;
        }
        sapeon_size_t GetNumLfcvts()
        {
            return kNumOfLfcvts;
        }

        sapeon_size_t GetNumDmaRegions()
        {
            return kNumOfDmaRegions;
        }

    protected:
        static constexpr sapeon_size_t kNumOfCores = 4;
        static constexpr sapeon_size_t kNumOfLfcvts = 4;
        static constexpr sapeon_size_t kNumOfDmaRegions = 4;

        virtual sapeon_result_t RegWrite(const sapeon_addr_t addr, const void *src,
                                         const sapeon_size_t size) = 0;
        virtual sapeon_result_t RegRead(void *dst, const sapeon_addr_t addr,
                                        const sapeon_size_t size) = 0;

        sapeon_size_t device_id() const { return kDeviceId_; }

    private:
        const sapeon_size_t kDeviceId_;

        std::mutex m_aicore_[kNumOfCores];
        std::mutex m_icvt_[kNumOfLfcvts];
        std::mutex m_ocvt_[kNumOfLfcvts];

        sapeon_addr_t get_lfcvt_reg_addr(const sapeon_size_t channel_id,
                                         const sapeon_size_t offset) const;
    };
} // namespace sapeon::emul
