/*
 * x340_rt_grpc.h
 *
 *  Author: Heecheol Yang (heecheol.yang@sapeon.com)
 *  Description:
 *   This file is a header file for X340 RT API based on X330 grpc client
 *  Copyright 2024 SAPEON Inc.
 *
 *  Revision History:
 *   - 2024.02.26 : initial version
 */
#ifndef LIB_C_X340_RT_GRPC_H__
#define LIB_C_X340_RT_GRPC_H__

// System includes

// Project

#include <api/c/x340_rt.h>
#include <types.h>
#include <string>
#ifdef __cplusplus
extern "C" {
#endif

// Type Declarations

struct x340_rt_api_grpc;

// Function Declarations
/**
 * @brief Create gRPC based API for X340 RT
 * @param device_id device id
 * @return If the operation is successful, return API instance. Otherwise,
 * return nullptr
 */
x340_rt_api_data_t* x340_rt_umd_asic_create(sapeon_size_t device_id);

// Function Declarations
/**
 * @brief Create gRPC based API for X340 RT
 * @param server_address Server address
 * @return If the operation is successful, return API instance. Otherwise,
 * return nullptr
 */
x340_rt_api_data_t* x340_rt_grpc_create(std::string server_address);

/**
 * @brief Release gRPC based API for X340 RT. Note that this function
 * invalidates api_data.
 * @param api_data API instance
 * @return Always return SAPEON_OK
 */
sapeon_result_t x340_rt_grpc_release(x340_rt_api_data_t* api_data);
#ifdef __cplusplus
}
#endif

#endif