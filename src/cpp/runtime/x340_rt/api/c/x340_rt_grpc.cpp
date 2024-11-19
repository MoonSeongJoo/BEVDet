/*
 * x340_rt_grpc.cpp
 *
 *  Author: Heecheol Yang (heecheol.yang@sapeon.com)
 *  Description:
 *   This file is a implementation of X340 RT API based on X330 grpc client
 *  Copyright 2024 SAPEON Inc.
 *
 *  Revision History:
 *   - 2024.02.26 : initial version
 */

// Main include
#include "x340_rt_grpc.h"

// System includes
// #include <memory>

// Project
#include <api/c/x340_rt.h>
#include <types.h>

#include <emulator/lib/emul.hpp>

// Type Definitions

/**
 * @brief gRPC based API for X340 RT
 * @details This API is used to communicate with X340 RT using gRPC
 */
typedef struct x340_rt_api_grpc {
  std::unique_ptr<sapeon::emul::X340toX330Emul> x340_emul;
} x340_rt_api_grpc;

// Function Definitions

/**
 * @brief Release gRPC based API for X340 RT. Note that this function
 * invalidates api_data.
 * @param api_data API instance
 * @return Always return SAPEON_OK
 */
sapeon_result_t x340_rt_grpc_release(x340_rt_api_data_t* api_data) {
  if (api_data == nullptr) {
    return SAPEON_OK;
  }
  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_OK;
  }
  priv->x340_emul.reset();
  delete priv;
  delete api_data;
  return SAPEON_OK;
}

/**
 * @brief Initialize gRPC based API for X340 RT
 * @param api_data API instance
 * @param server_address Server address
 * @return Result of the operation
 */
static sapeon_result_t x340_rt_grpc_init(x340_rt_api_data_t* api_data) {
  return SAPEON_OK;
}

/**
 * @brief Write data to device DRAM
 * @param api_data API instance
 * @param dst
 * @param src
 * @param size
 * @return sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_dma_write(x340_rt_api_data_t* api_data,
                                       const sapeon_addr_t dst,
                                       const sapeon_byte_t* src,
                                       sapeon_size_t size) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }
  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }
  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }
  return priv->x340_emul->DmaWrite(dst, src, size);
}

/**
 * @brief Read data from device DRAM
 * @param api_data API instance
 * @param dst
 * @param src
 * @param size
 * @return sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_dma_read(x340_rt_api_data_t* api_data,
                                      sapeon_byte_t* dst,
                                      const sapeon_addr_t src,
                                      sapeon_size_t size) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  auto result = priv->x340_emul->DmaRead(dst, src, size);
  if (result != SAPEON_OK) {
    return result;
  }
  return result;
}

/**
 * @brief Run CPS on the specified core
 * @param api_data API instance
 * @param core_id
 * @param cps_addr
 * @param cps_size
 * @param dma_regions
 * @param dma_regions_size
 * @return sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_run_inference(
    x340_rt_api_data_t* api_data, const sapeon_size_t core_id,
    const sapeon_addr_t cps_addr, const sapeon_addr_t cps_size,
    const sapeon_addr_t dma_regions[], const sapeon_size_t dma_regions_size) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  std::vector<sapeon_addr_t> dma_regions_vec(dma_regions,
                                             dma_regions + dma_regions_size);
  return priv->x340_emul->RunInference(core_id, cps_addr, cps_size,
                                       dma_regions_vec);
}

/**
 * @brief Run ICVT on the specified core
 * @param api_data API instance
 * @param icvt_id
 * @param cmd
 * @return * sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_run_icvt(x340_rt_api_data_t* api_data,
                                      const sapeon_size_t icvt_id,
                                      const ICvtCommand* cmd) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  return priv->x340_emul->RunIcvt(icvt_id, *cmd);
}

/**
 * @brief Run OCVT on the specified core
 * @param api_data API instance
 * @param ocvt_id
 * @param cmd
 * @return sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_run_ocvt(x340_rt_api_data_t* api_data,
                                      const sapeon_size_t ocvt_id,
                                      const OCvtCommand* cmd) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  return priv->x340_emul->RunOcvt(ocvt_id, *cmd);
}

/**
 * @brief Set core group ID for the specified core
 * @param api_data API instance
 * @param core_id
 * @param gid
 * @return sapeon_result_t
 */
sapeon_result_t x340_rt_grpc_set_gid(x340_rt_api_data_t* api_data,
                                     const sapeon_size_t core_id,
                                     const sapeon_size_t gid) {
  if (api_data == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  if (priv == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  if (priv->x340_emul == nullptr) {
    return SAPEON_INVALID_ARGUMENT;
  }

  return priv->x340_emul->SetGid(core_id, gid);
}

x340_rt_api_data_t* x340_rt_umd_asic_create(sapeon_size_t device_id) {
  x340_rt_api_data_t* api_data = new x340_rt_api_data_t();
  if (api_data == nullptr) {
    return nullptr;
  }
  api_data->priv = new x340_rt_api_grpc();
  if (api_data->priv == nullptr) {
    delete api_data;
    return nullptr;
  }
  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  priv->x340_emul =
      std::make_unique<sapeon::emul::X340toX330Emul>(device_id);
  if(SAPEON_OK == priv->x340_emul->GetDeviceStatus())
  {
  api_data->Init = x340_rt_grpc_init;
  api_data->Release = x340_rt_grpc_release;
  api_data->DmaWrite = x340_rt_grpc_dma_write;
  api_data->DmaRead = x340_rt_grpc_dma_read;
  api_data->RunIcvt = x340_rt_grpc_run_icvt;
  api_data->RunOcvt = x340_rt_grpc_run_ocvt;
  api_data->RunInference = x340_rt_grpc_run_inference;
  api_data->SetGid = x340_rt_grpc_set_gid;
  }
  else{
    api_data = nullptr;
  }
  return api_data;
}


x340_rt_api_data_t* x340_rt_grpc_create(std::string server_address) {
  x340_rt_api_data_t* api_data = new x340_rt_api_data_t();
  if (api_data == nullptr) {
    return nullptr;
  }
  api_data->priv = new x340_rt_api_grpc();
  if (api_data->priv == nullptr) {
    delete api_data;
    return nullptr;
  }
  x340_rt_api_grpc* priv = reinterpret_cast<x340_rt_api_grpc*>(api_data->priv);
  priv->x340_emul =
      std::make_unique<sapeon::emul::X340toX330Emul>(server_address);
  if(SAPEON_OK == priv->x340_emul->GetDeviceStatus())
  {
  api_data->Init = x340_rt_grpc_init;
  api_data->Release = x340_rt_grpc_release;
  api_data->DmaWrite = x340_rt_grpc_dma_write;
  api_data->DmaRead = x340_rt_grpc_dma_read;
  api_data->RunIcvt = x340_rt_grpc_run_icvt;
  api_data->RunOcvt = x340_rt_grpc_run_ocvt;
  api_data->RunInference = x340_rt_grpc_run_inference;
  api_data->SetGid = x340_rt_grpc_set_gid;
  }
  else{
    api_data = nullptr;
  }

  return api_data;
}
