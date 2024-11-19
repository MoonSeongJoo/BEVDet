#include <types.h>
#include <vector>
#include <emulator/x330_umd/x330_regmap.h>

namespace sapeon::emul
{
    class X330Client {
    public:
    X330Client() = default;
    virtual sapeon_result_t DmaWrite(const sapeon_addr_t dst,
                                    const sapeon_byte_t* src,
                                    sapeon_size_t size) = 0;

    virtual sapeon_result_t DmaRead(sapeon_byte_t *dst,
                                    const sapeon_addr_t src,
                                    sapeon_size_t size) = 0;
    virtual sapeon_result_t RunInference(
        const sapeon_size_t core_id, const sapeon_addr_t cps_addr,
        const sapeon_addr_t cps_size,
        const std::vector<sapeon_addr_t>& dma_regions) = 0;
    virtual sapeon_result_t RunIcvt(const sapeon_size_t icvt_id,
                                    const ICvtCommand& cmd) = 0;
    virtual sapeon_result_t RunOcvt(const sapeon_size_t ocvt_id,
                                    const OCvtCommand& cmd) = 0;
    virtual sapeon_result_t SetGid(const sapeon_size_t core_id,
                                    const sapeon_size_t gid) = 0;

    private:
    };
}
