#pragma once
#ifdef __cplusplus
extern "C" {
#endif

// Standard includes

// System includes
#include <semaphore.h>
#include <stdbool.h>
#include <stddef.h>
// RT
#include <types.h>

//------------------------------------------------------------------------------
// Address Encoding and Alignment
//------------------------------------------------------------------------------
const u32 DDR_ADDRESS_ENCODING_SHIFT = 5;

/*----------------------------------------------------------------------------
 * Data type code
 *----------------------------------------------------------------------------*/
enum DataType {
  DTY_I8 = 0,
  DTY_I8U = 1,
  DTY_I10U = 2,
  DTY_NF8 = 3,
  DTY_NF8U = 4,
  DTY_NF8E = 5,
  DTY_NF10 = 6,
  DTY_NF10U = 7,
  DTY_NF16 = 8,
  DTY_BF16 = 9,
  DTY_FP32 = 10,
  DTY_INVALID = 16
};

#define MXC_HEIGHT (128)
#define MXC_WIDTH (64)
#define CORE_COUNT (4)
#define LFCVT_COUNT (4)

#define DDR_ADDRESS_SECTION_BASE (0x200000000ULL)
#define CORE_ADDRESS_SECTION_BASE (0x040000000ULL)
#define LFCVT_ADDRESS_SECTION_BASE (0x040002000ULL)
#define DDR_ADDRESS_SUBSECT_COUNT (2)
#define DDR_ADDRESS_SUBSECT_STRIDE (1024ULL * 1024 * 1024 * 8)
#define DDR_ADDRESS_SUBSECT_SIZE (1024ULL * 1024 * 1024 * 8)  // clamshell
#define CORE_ADDRESS_SIZE (2 * 1024ULL)
#define LFCVT_ADDRESS_SIZE (2 * 1024ULL)

/*****************************************************/
/* DCS */
/* Register Offsets */

#define DCS_REG_GLOBAL_DESC (0x00)
#define DCS_REG_GLOBAL_DESC_CFG (0x00)
#define DCS_REG_GLOBAL_DESC_FINISH_TIMEOUT (0x04)
#define DCS_REG_GLOBAL_DESC_ABORT_TIMEOUT (0x08)
#define DCS_REG_GLOBAL_DESC_MXC_GRAD_ACT (0x0C)
#define DCS_REG_FORCE_ABORT (0x14)

#define DCS_REG_BYPASS_CFG (0x20)
#define DCS_REG_BYPASS_CMD (0x28)

#define DCS_REG_NEXT_JOB_DESC (0x30)
#define DCS_REG_NEXT_JOB_DESC_HOOK_TABLE (0x30)
#define DCS_REG_NEXT_JOB_DESC_LOG_ADDRESS (0x38)
#define DCS_REG_NEXT_JOB_DESC_CPS_ADDRESS (0x3C)
#define DCS_REG_NEXT_JOB_DESC_CPS_LENGTH (0x40)
#define DCS_REG_NEXT_JOB_DESC_DMA_REGIONS (0x44)
#define DCS_REG_NEXT_JOB_DESC_CFG (0x54)

#define DCS_REG_CURR_JOB_DESC (0x60)
#define DCS_REG_CURR_JOB_DESC_HOOK_TABLE (0x60)
#define DCS_REG_CURR_JOB_DESC_LOG_ADDRESS (0x68)
#define DCS_REG_CURR_JOB_DESC_CPS_ADDRESS (0x6C)
#define DCS_REG_CURR_JOB_DESC_CPS_LENGTH (0x70)
#define DCS_REG_CURR_JOB_DESC_DMA_REGIONS (0x74)
#define DCS_REG_CURR_JOB_DESC_CFG (0x84)

#define DCS_REG_STATUS_RO (0x90)
#define DCS_REG_STATUS_AC (0x94)

#define DCS_REG_PERF_CLEAR (0xA0)
#define DCS_REG_FREE_RUN_CYCLE (0xA4)
#define DCS_REG_DCS_BUSY_CYCLE (0xA8)
#define DCS_REG_EVE_BUSY_CYCLE (0xAC)
#define DCS_REG_UDR_BUSY_CYCLE (0xB0)
#define DCS_REG_LDR_BUSY_CYCLE (0xB4)
#define DCS_REG_LDT_BUSY_CYCLE (0xB8)
#define DCS_REG_MXC_BUSY_CYCLE (0xBC)
#define DCS_REG_NVP_BUSY_CYCLE (0xC0)
#define DCS_REG_DCS_CPS_COUNT (0xC4)
#define DCS_REG_EVE_CMD_COUNT (0xC8)
#define DCS_REG_UDR_CMD_COUNT (0xCC)
#define DCS_REG_LDR_CMD_COUNT (0xD0)
#define DCS_REG_LDT_CMD_COUNT (0xD4)
#define DCS_REG_MXC_CMD_COUNT (0xD8)
#define DCS_REG_NVP_CMD_COUNT (0xDC)

#define DCS_REG_TCACHE_SEL (0x110)
#define DCS_REG_TCACHE_WREQ (0x114)
#define DCS_REG_TCACHE_RREQ (0x118)
#define DCS_REG_TCACHE_REND (0x11C)
#define DCS_REG_TCACHE_DATA (0x400)

/*****************************************************/
/** Utility functions **/
u64 DecodeDdrAddress(u32 enc_address);
u32 EncodeDdrAddress(u64 raw_address);
int IDivCeil(int x, int y);
int ICeil(int x, int y);
int IFllor(int x, int y);
int sizeof_dtype(int dty, bool ceiling);
int bit_sizeof_dtype(int dty);
const char* nameof_dtype(int dty);

/** End of Utility functions **/
/*****************************************************/

/*----------------------------------------------------------------------------
 * DCS Register Structures
 *----------------------------------------------------------------------------*/
union DcsRegGlobalDescConfig {
  u32 raw_u32;
  struct {
    u32 finish_intr_enable : 1;
    u32 finish_intr_dst : 1;
    u32 failure_intr_enable : 1;
    u32 failure_intr_dst : 1;
    u32 eve_intr_enable : 1;
    u32 eve_intr_dst : 1;
    u32 icsync_group_id : 2;
    u32 __reserved__ : 24;
  };
};

union DcsRegGlobalDescMxcGradAct {
  u32 raw_u32[2];
  struct {
    u32 up_tracking_only : 1;
    u32 dn_tracking_only : 1;
    u32 dn_steps_on_alert : 5;
    u32 min_active_level : 5;
    u32 alert_det_cycles : 8;
    u32 backp_det_cycles : 8;
    u32 disable_clk_off : 1;
    u32 __reserved__ : 3;
    u16 up_step_cycles;
    u16 dn_step_cycles;
  };
};

#pragma pack(push, 4)  // for tight packing
struct DcsRegGlobalDesc {
  union DcsRegGlobalDescConfig cfg;
  u32 finish_timeout;
  u32 abort_timeout;
  union DcsRegGlobalDescMxcGradAct mxc_grad_act;
};
#pragma pack(pop)

union DcsRegBypassConfig {
  u32 raw_u32;
  struct {
    u32 payload_len : 14;
    u32 cxu_id : 3;
    u32 __reserved__ : 15;
  };
};

union DcsRegJobDesc_Config {
  u32 raw_u32;
  struct {
    u32 valid : 1;
    u32 barrier_all : 1;
    u32 log_enable : 1;
    u32 cps_rd_qos : 4;
    u32 log_wr_qos : 4;
    u32 abort_on_cps_rd_error : 1;
    u32 abort_on_log_wr_error : 1;
    u32 abort_on_udr_error : 1;
    u32 abort_on_ldr_error : 1;
    u32 abort_on_ldt_error : 1;
    u32 abort_on_mxc_error : 1;
    u32 abort_on_nvp_error : 1;
    u32 __reserved__ : 14;
  };
};

#pragma pack(push, 4)  // for tight packing
struct DcsRegJobDesc {
  u64 hook_table;
  u32 log_address;
  u32 cps_address;
  u32 cps_length;
  u32 dma_regions[4];
  union DcsRegJobDesc_Config cfg;
};
#pragma pack(pop)

enum DCS_STATUS {
  DCS_NONE = 0,
  DCS_EVE_EVENT = 1,
  DCS_CPS_FINISH = 2,
  DCS_CPS_FAILURE = 3
};
union DcsRegStatus {
  u32 raw_u32;
  struct {
    u32 busy : 1;            // RO
    u32 next_valid : 1;      // RO
    u32 bypass_busy : 1;     // RO
    u32 abort_busy : 1;      // RO
    u32 event_type : 2;      // RO/AC
    u32 eve_event_id : 16;   // RO
    u32 finish_timeout : 1;  // RO/AC
    u32 abort_timeout : 1;   // RO/AC
    u32 checksum_error : 1;  // RO/AC
    u32 cps_rd_error : 1;    // RO/AC
    u32 log_wr_error : 1;    // RO/AC
    u32 udr_error : 1;       // RO/AC
    u32 ldr_error : 1;       // RO/AC
    u32 ldt_error : 1;       // RO/AC
    u32 mxc_error : 1;       // RO/AC
    u32 nvp_error : 1;       // RO/AC
  };
};

/*----------------------------------------------------------------------------
 * DCS Log entry
 *----------------------------------------------------------------------------*/
enum DCS_LOG { DCS_LOG_START = 0, DCS_LOG_RDONE = 1, DCS_LOG_WDONE = 2 };
union DcsLogEntry {
  u64 raw_u64;
  struct {
    u16 event : 2;
    u16 is_cxu : 1;
    u16 cxu_id : 3;
    u16 __reserved__ : 10;
    u16 index;
    u32 cycle;
  };
};

/*****************************************************/
/* LFCVT */
const u32 LFCVT_REG_OCVT_STATUS_RO = 1024 * 0 + 4 * 0;
const u32 LFCVT_REG_OCVT_STATUS_AC = 1024 * 0 + 4 * 1;
const u32 LFCVT_REG_OCVT_PUSH_CMD = 1024 * 0 + 4 * 2;
const u32 LFCVT_REG_OCVT_NEXT_CMD = 1024 * 0 + 4 * 3;
const u32 LFCVT_REG_ICVT_STATUS_RO = 1024 * 1 - 4 * (14 + 3);
const u32 LFCVT_REG_ICVT_STATUS_AC = 1024 * 1 - 4 * (14 + 2);
const u32 LFCVT_REG_ICVT_PUSH_CMD = 1024 * 1 - 4 * (14 + 1);
const u32 LFCVT_REG_ICVT_NEXT_CMD = 1024 * 1 - 4 * (14 + 0);

const int LFCVT_SUBCH_COUNT = CORE_COUNT / LFCVT_COUNT;
const int LFCVT_SUBCH_ADDRESS_SIZE = 2 * 1024;

union LfcvtRegStatus {
  u32 raw_u32;
  struct {
    u32 next_valid : 1;   // RO
    u32 crnt_valid : 1;   // RO
    u32 crnt_busy : 1;    // RO
    u32 crnt_finish : 1;  // RO
    u32 crnt_rd_err : 1;  // RO
    u32 crnt_wr_err : 1;  // RO
    u32 prev_finish : 1;  // RO/AC
    u32 prev_rd_err : 1;  // RO/AC
    u32 prev_wr_err : 1;  // RO/AC
    u32 __reserved__ : 23;
  };
};

struct LfcvtCommand {};
static int Lfcvt_EncodeChOrder(int c0, int c1, int c2, int c3) {
  int v = 0;
  v |= c0 << (2 * 0);
  v |= c1 << (2 * 1);
  v |= c2 << (2 * 2);
  v |= c3 << (2 * 3);
  return v;
}
enum LfcvtCh {
  LFCVT_CH_E1 = 1,
  LFCVT_CH_E2 = 2,
  LFCVT_CH_E3 = 3,
  LFCVT_CH_E4 = 4,
  LFCVT_CH_EX = 0
};
#pragma pack(push, 4)  // for tight packing
struct ICvtCommand {
  u32 __reserved__ : 3;
  u32 has_next_chain : 1;
  u32 in_data_type : 4;
  u32 in_exp_bias : 6;
  u32 out_fblk_rate : 3;
  u32 out_fblk_vpack : 1;
  u32 out_fblk_linear : 1;
  u32 out_ch_mode : 3;
  u32 out_data_type : 4;
  u32 out_exp_bias : 6;

  u8 bus_rd_qos : 4;
  u8 bus_wr_qos : 4;
  u8 out_ch_order;
  u8 out_phybatch_size;
  u8 in_image_count;

  u8 out_cluster_size;
  u8 out_fblk_count;
  u16 in_image_width;

  u16 out_fblk_height;
  u16 out_ch_length;

  u32 in_stride_y;
  u32 out_stride_p;
  u32 out_stride_x;
  u32 out_stride_y;
  u16 out_ch_scale[4];
  u16 out_ch_bias[4];

  u64 out_address;
  u64 in_address[MXC_HEIGHT];
};
#pragma pack(pop)

int GetInImageHeight(const struct ICvtCommand* cmd);
int GetInChannels(const struct ICvtCommand* cmd);
int GetInLineSize(const struct ICvtCommand* cmd);
int GetIcvtCmdSize(const struct ICvtCommand* cmd);

#pragma pack(push, 4)  // for tight packing
struct OCvtCommand {
  u32 __reserved__ : 4;
  u32 has_next_chain : 1;
  u32 out_data_type : 4;
  u32 out_exp_bias : 6;
  u32 in_fblk_rate : 3;
  u32 in_fblk_linear : 1;
  u32 in_ch_mode : 3;
  u32 in_data_type : 4;
  u32 in_exp_bias : 6;

  u8 bus_rd_qos : 4;
  u8 bus_wr_qos : 4;
  u8 out_ch_order;
  u8 out_image_count;
  u8 in_phybatch_size;

  u8 in_cluster_size;
  u8 in_fblk_count;
  u16 out_image_width;

  u16 in_fblk_height;
  u16 in_ch_length;

  u32 in_stride_p;
  u32 in_stride_x;
  u32 in_stride_y;
  u32 out_stride_i;
  u32 out_stride_y;
  u16 out_ch_scale[4];
  u16 out_ch_bias[4];

  u64 out_address;
  u64 in_address;
};
#pragma pack(pop)

int GetOutImageHeight(const struct OCvtCommand* cmd);
int GetOutChannels(const struct OCvtCommand* cmd);
int GetOutLineSize(const struct OCvtCommand* cmd);
int GetOcvtCmdSize();

/* End of LFCVT */
/*****************************************************/

#ifdef __cplusplus
}
#endif