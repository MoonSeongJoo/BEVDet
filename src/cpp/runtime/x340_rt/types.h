#ifndef TYPES_H
#define TYPES_H

// Standard includes
#include <stdbool.h>
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
// System includes

// RT

/*************************************/
/* Primitive types                   */

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t s8;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

/*************************************/

/*************************************/
/* Sapeon types                      */

/* Primitive Sapeon types */

typedef u64 sapeon_addr_t;
typedef u8 sapeon_byte_t;
typedef u64 sapeon_size_t;

/* Complex Sapeon types */

typedef enum {
  SAPEON_OK = 0,
  SAPEON_NG,
  SAPEON_INVALID_DMA_REGION_SIZE,       // # of DMA Regions must be 4
  SAPEON_CPS_ADDR_NOT_ALIGNED,          // 5 LSBs of CPS address must be 0
  SAPEON_CPS_SIZE_NOT_ALIGNED,          // CPS size must be multiple of 8
  SAPEON_DMA_REGION_NOT_ALIGNED,        // 5 LSBs of DMA region address must be 0
  SAPEON_INFERENCE_ERROR_BAD_CHECKSUM,  // Checksum error while running inference
  SAPEON_INFERENCE_ERROR_READ_ERROR,    // Read error while running inference
  SAPEON_DEVICE_OPEN_FAILURE,           // Failed to open device
  SAPEON_DEVICE_MMAP_FAILURE,           // Failed to mmap device
  SAPEON_SET_GID_FAILURE,               // Failed to set GID
  SAPEON_DMA_H2D_FAILURE,               // Failed to H2D DMA
  SAPEON_DMA_D2H_FAILURE,               // Failed to D2H DMA

  /*************************/
  /* Register Write Failures */
  SAPEON_REG_WRITE_FAILURE_GLOBAL_DESC_CFG,
  SAPEON_REG_WRITE_FAILURE_NEXT_JOB_DESC,
  SAPEON_REG_WRITE_FAILURE_STATUS_AC,
  SAPEON_REG_WRITE_FAILURE_GLOBAL_DESC_MXC_GRAD_ACT,
  /*************************/

  /*************************/
  /* Register Read Failures */
  SAPEON_REG_READ_FAILURE_STATUS_AC,
  /*************************/

  /*************************/
  /* Queue API Status*/
  SAPEON_QUEUE_EMPTY,
  SAPEON_QUEUE_FULL,
  SAPEON_QUEUE_LOCK_ACQUIRE_FAILED,

  /*************************/
  /* Lock API Status*/
  SAPEON_LOCK_ACQUIRE_FAILED,

  SAPEON_INVALID_ARGUMENT,
} sapeon_result_t;

/*************************************/

#endif  // TYPES_H