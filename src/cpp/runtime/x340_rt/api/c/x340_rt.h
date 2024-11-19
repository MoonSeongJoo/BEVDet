/*
 * x340_rt.h
 *
 *  Author: Heecheol Yang (heecheol.yang@sapeon.com)
 *  Description:
 *   This file is a header file for X340 RT API
 *  Copyright  2024 SAPEON Inc.
 *
 *  Revision History:
 *   - 2024.02.26 : initial version
 */
#ifndef LIB_C_X340_RT_H__
#define LIB_C_X340_RT_H__

#ifdef __cplusplus
extern "C" {
#endif

// System includes

// Project

// Todo(Heecheol): Change the included header file to X340 files
#include <emulator/x330_umd/x330_regmap.h>
#include <types.h>

// Type Definitions

/**
 * @brief Abstract API for X340 RT
 * @details This API is used to communicate with X340 RT
 */
typedef struct _x340_rt_api_data {
  /**
   * @brief Private data
   *
   */
  void* priv;
  /**
   * @brief Initialize API for X340 RT
   * @param api_data API instance
   * @return Result of the operation
   */
  sapeon_result_t (*Init)(struct _x340_rt_api_data* api_data);

  /**
   * @brief Release API for X340 RT
   * @param api_data API instance
   * @return Result of the operation
   *
   */
  sapeon_result_t (*Release)(struct _x340_rt_api_data* api_data);

  /**
   * @brief Write data to device DRAM
   * @param api_data API instance
   * @param dst Destination address in device DRAM
   * @param src Source data
   * @param size Size of source data
   * @return Result of the operation
   */
  sapeon_result_t (*DmaWrite)(struct _x340_rt_api_data* api_data, const sapeon_addr_t dst,
                              const sapeon_byte_t* src, sapeon_size_t size);
  /**
   * @brief Read data from device DRAM
   * @param api_data API instance
   * @param dst Destination buffer in host memory
   * @param src Source address in device DRAM
   * @param size Size of source data
   * @return Result of the operation
   */
  sapeon_result_t (*DmaRead)(struct _x340_rt_api_data* api_data, sapeon_byte_t* dst,
                             const sapeon_addr_t src, sapeon_size_t size);

  /**
   * @brief Run CPS on the specified core
   * @param api_data API instance
   * @param core_id Core ID
   * @param cps_addr Address of CPS in device DRAM
   * @param cps_size Size of CPS
   * @param dma_regions Array of DMA regions
   * @param dma_regions_size Size of DMA regions
   *
   */
  sapeon_result_t (*RunInference)(struct _x340_rt_api_data* api_data, const sapeon_size_t core_id,
                                  const sapeon_addr_t cps_addr, const sapeon_addr_t cps_size,
                                  const sapeon_addr_t dma_regions[],
                                  const sapeon_size_t dma_regions_size);
  /**
   * @brief Run ICVT on the specified core
   * @param api_data API instance
   * @param icvt_id ICVT ID
   * @param cmd ICVT command
   *
   */
  sapeon_result_t (*RunIcvt)(struct _x340_rt_api_data* api_data, const sapeon_size_t icvt_id,
                             const struct ICvtCommand* cmd);
  /**
   * @brief Run OCVT on the specified core
   * @param api_data API instance
   * @param ocvt_id OCVT ID
   * @param cmd OCVT command
   *
   */
  sapeon_result_t (*RunOcvt)(struct _x340_rt_api_data* api_data, const sapeon_size_t ocvt_id,
                             const struct OCvtCommand* cmd);
  /**
   * @brief Set core group ID for the specified core
   * @param api_data API instance
   * @param core_id Core ID
   * @param gid Group ID
   *
   */
  sapeon_result_t (*SetGid)(struct _x340_rt_api_data* api_data, const sapeon_size_t core_id,
                            const sapeon_size_t gid);

} x340_rt_api_data_t;

#ifdef __cplusplus
}
#endif

#endif

/** \example example_yolov2_main.c
 * Simple example of how to use the X340 RT API to run YOLOv2 on X340
 */