// Main header
#include "x330_regmap.h"

u64 DecodeDdrAddress(u32 enc_address) {
  return u64(enc_address) << DDR_ADDRESS_ENCODING_SHIFT;
}

u32 EncodeDdrAddress(u64 raw_address) {
  u32 enc_address = raw_address >> DDR_ADDRESS_ENCODING_SHIFT;
  return enc_address;
}

template <typename T>
T DivCeil(T x, T y) {
  return (x + y - 1) / y;
}

template <typename T>
T Ceil(T x, T y) {
  return DivCeil(x, y) * y;
}

template <typename T>
T Floor(T x, T y) {
  return (x / y) * y;
}

int IDivCeil(int x, int y) { return DivCeil(x, y); }
int ICeil(int x, int y) { return Ceil(x, y); }
int IFllor(int x, int y) { return Floor(x, y); }

int sizeof_dtype(int dty, bool ceiling) {
  switch (dty) {
    case DTY_I8:
      return 1;
    case DTY_I8U:
      return 1;
    case DTY_I10U:
      return ceiling ? 2 : 1;
    case DTY_NF8:
      return 1;
    case DTY_NF8U:
      return 1;
    case DTY_NF8E:
      return 1;
    case DTY_NF10:
      return ceiling ? 2 : 1;
    case DTY_NF10U:
      return ceiling ? 2 : 1;
    case DTY_NF16:
      return 2;
    case DTY_BF16:
      return 2;
    case DTY_FP32:
      return 4;
    default:
      return -1;
  }
}

int bit_sizeof_dtype(int dty) {
  switch (dty) {
    case DTY_I8:
      return 8;
    case DTY_I8U:
      return 8;
    case DTY_I10U:
      return 10;
    case DTY_NF8:
      return 8;
    case DTY_NF8U:
      return 8;
    case DTY_NF8E:
      return 8;
    case DTY_NF10:
      return 10;
    case DTY_NF10U:
      return 10;
    case DTY_NF16:
      return 16;
    case DTY_BF16:
      return 16;
    case DTY_FP32:
      return 32;
    default:
      return -1;
  }
}

const char* nameof_dtype(int dty) {
  switch (dty) {
    case DTY_I8:
      return "I8";
    case DTY_I8U:
      return "I8U";
    case DTY_I10U:
      return "I10U";
    case DTY_NF8:
      return "NF8";
    case DTY_NF8U:
      return "NF8U";
    case DTY_NF8E:
      return "NF8E";
    case DTY_NF10:
      return "NF10";
    case DTY_NF10U:
      return "NF10U";
    case DTY_NF16:
      return "NF16";
    case DTY_BF16:
      return "BF16";
    case DTY_FP32:
      return "FP32";
    default:
      return "INVALID";
  }
}

int GetInImageHeight(const struct ICvtCommand* cmd) {
  int Ofbh = cmd->out_fblk_height;
  int Ofbc = cmd->out_fblk_count;
  return Ofbh * Ofbc;
}

int GetInChannels(const struct ICvtCommand* cmd) {
  int Ofbr = cmd->out_fblk_rate;
  int Ic = cmd->out_ch_length;
  switch (cmd->out_ch_mode) {
    case LFCVT_CH_E1:
      Ic *= 1;
      break;
    case LFCVT_CH_E2:
      Ic *= 2;
      break;
    case LFCVT_CH_E3:
      Ic *= 3;
      break;
    case LFCVT_CH_E4:
      Ic *= 4;
      break;
    case LFCVT_CH_EX:
      if (sizeof_dtype(cmd->out_data_type, false) == 1) {
        Ic *= 8 << Ofbr;
      } else {
        Ic *= 4 << Ofbr;
      }
      break;
    default:
      return SAPEON_NG;
  }
  return Ic;
}

int GetInLineSize(const struct ICvtCommand* cmd) {
  int Ic = GetInChannels(cmd);
  int in_line_size = cmd->in_image_width;
  int in_elem_bits = bit_sizeof_dtype(cmd->in_data_type);
  if (in_elem_bits == 8)
    in_line_size *= Ic;
  else if (in_elem_bits == 10)
    in_line_size *= 4;
  else if (in_elem_bits == 16)
    in_line_size *= Ic * 2;
  else
    in_line_size *= Ic * 4;
  return in_line_size;
}

int GetIcvtCmdSize(const struct ICvtCommand* cmd) {
  return offsetof(struct ICvtCommand, in_address) +
         sizeof(u64) * cmd->in_image_count;
}

int GetOutImageHeight(const struct OCvtCommand* cmd) {
  int Ifbh = cmd->in_fblk_height;
  int Ifbc = cmd->in_fblk_count;
  return Ifbh * Ifbc;
}

int GetOutChannels(const struct OCvtCommand* cmd) {
  int Ifbr = cmd->in_fblk_rate;
  int Oc = cmd->in_ch_length;
  switch (cmd->in_ch_mode) {
    case LFCVT_CH_E1:
      Oc *= 1;
      break;
    case LFCVT_CH_E2:
      Oc *= 2;
      break;
    case LFCVT_CH_E3:
      Oc *= 3;
      break;
    case LFCVT_CH_E4:
      Oc *= 4;
      break;
    case LFCVT_CH_EX:
      if (sizeof_dtype(cmd->in_data_type, false) == 1) {
        Oc *= 8 << Ifbr;
      } else {
        Oc *= 4 << Ifbr;
      }
      break;
    default:
      return SAPEON_NG;
  }
  return Oc;
}

int GetOutLineSize(const struct OCvtCommand* cmd) {
  int Oc = GetOutChannels(cmd);
  int out_line_size = cmd->out_image_width;
  int out_elem_bits = bit_sizeof_dtype(cmd->out_data_type);
  if (out_elem_bits == 8)
    out_line_size *= Oc;
  else if (out_elem_bits == 10)
    out_line_size *= 4;
  else if (out_elem_bits == 16)
    out_line_size *= Oc * 2;
  else
    out_line_size *= Oc * 4;

  int mem_unit = 32;
  out_line_size = ICeil(out_line_size, mem_unit);
  return out_line_size;
}

int GetOcvtCmdSize() { return sizeof(struct OCvtCommand); }