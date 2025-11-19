#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"

const uint16_t kLogLut[] = {
    0,    224,  442,  654,  861,  1063, 1259, 1450, 1636, 1817, 1992, 2163,
    2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
    3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
    5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
    5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
    5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
    4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
    2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
    1094, 963,  830,  695,  559,  421,  282,  142,  0,    0};

inline vuint32m4_t MulHighFixedPoint16_UU_vx_u32m4(vuint32m4_t v_a, uint32_t b, size_t vl)
{
  // load scalar and perform low and high multiplication
  vuint32m4_t v_b = __riscv_vmv_v_x_u32m4(b, vl);
  vuint32m4_t v_lo = __riscv_vmul_vv_u32m4(v_a, v_b, vl);
  vuint32m4_t v_hi = __riscv_vmulhu_vv_u32m4(v_a, v_b, vl);

  // Add rounding constant 32768 to the low part
  vuint32m4_t v_round = __riscv_vmv_v_x_u32m4(32768, vl);
  vuint32m4_t v_lo_rounded = __riscv_vadd_vv_u32m4(v_lo, v_round, vl);

  // Detect carry from the low part addition and propagate to high part
  vbool8_t v_carry = __riscv_vmsltu_vv_u32m4_b8(v_lo_rounded, v_lo, vl);
  v_hi = __riscv_vadd_vx_u32m4_m(v_carry, v_hi, 1, vl);

  // Combine high shifted left and low shifted right
  return __riscv_vor_vv_u32m4(
      __riscv_vsll_vx_u32m4(v_hi, 16, vl),
      __riscv_vsrl_vx_u32m4(v_lo_rounded, 16, vl), vl);
}

inline vint32m4_t MulHighFixedPoint16_SU_vx_i32m4(vuint32m4_t v_unsigned, int32_t signed_scalar, size_t vl)
{
  // Load signed scalar and perform low and high multiplication
  vint32m4_t v_signed_scalar = __riscv_vmv_v_x_i32m4(signed_scalar, vl);
  vint32m4_t v_lo = __riscv_vmul_vv_i32m4(v_signed_scalar, __riscv_vreinterpret_v_u32m4_i32m4(v_unsigned), vl);
  vint32m4_t v_hi = __riscv_vmulhsu_vv_i32m4(v_signed_scalar, v_unsigned, vl);

  // Add rounding constant 32768 to the low part
  vint32m4_t v_round = __riscv_vmv_v_x_i32m4(32768, vl);
  vint32m4_t v_lo_rounded = __riscv_vadd_vv_i32m4(v_lo, v_round, vl);

  // Detect carry treating low part as unsigned and propagate to high part
  vbool8_t v_carry = __riscv_vmsltu_vv_u32m4_b8(
      __riscv_vreinterpret_v_i32m4_u32m4(v_lo_rounded),
      __riscv_vreinterpret_v_i32m4_u32m4(v_lo), vl);
  v_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_hi, 1, vl);

  // Combine high shifted left and low shifted right
  vuint32m4_t v_lo_shifted = __riscv_vsrl_vx_u32m4(
      __riscv_vreinterpret_v_i32m4_u32m4(v_lo_rounded), 16, vl);

  return __riscv_vor_vv_i32m4(
      __riscv_vsll_vx_i32m4(v_hi, 16, vl),
      __riscv_vreinterpret_v_u32m4_i32m4(v_lo_shifted), vl);
}

inline vuint32m4_t VectorLog2Int_u32m4(vuint32m4_t v_in, size_t vl)
{
  // Initialize result vector to zero
  vuint32m4_t v_result = __riscv_vmv_v_x_u32m4(0, vl);
  vuint32m4_t v_tmp;
  vbool8_t v_mask;

  // Check bit 16 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 16, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_m(v_mask, v_result, 16, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 8 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 8, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_m(v_mask, v_result, 8, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 4 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 4, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_m(v_mask, v_result, 4, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 2 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 2, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_m(v_mask, v_result, 2, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 1 and update result
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 1, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_m(v_mask, v_result, 1, vl);

  return v_result;
}

void FilterbankLogRVV(const uint32_t* input, int num_channels,
                      int32_t output_scale, uint32_t correction_bits,
                      int16_t* output)
{
  const uint32_t kLogScaleLog2 = 16;
  const uint32_t kLogCoeff = 45426;

  int i = 0;
  while (i < num_channels)
  {
    // Set vector length for 32-bit elements and group multiplier 4
    size_t vl = __riscv_vsetvl_e32m4(num_channels - i);

    // Load input, shift by correction bits, and determine active elements
    vuint32m4_t v_input = __riscv_vle32_v_u32m4(input + i, vl);
    vuint32m4_t v_scaled = __riscv_vsll_vx_u32m4(v_input, correction_bits, vl);
    vbool8_t v_active = __riscv_vmsgtu_vx_u32m4_b8(v_scaled, 1, vl);

    // Calculate integer part of log2
    vuint32m4_t v_integer = VectorLog2Int_u32m4(v_scaled, vl);

    // Calculate shift amount to align MSB to bit 16
    vint32m4_t v_shift_amt = __riscv_vrsub_vx_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(v_integer), 16, vl);

    // Create mask for left shifting vs right shifting
    vbool8_t v_shift_left_mask = __riscv_vmsgt_vx_i32m4_b8(v_shift_amt, 0, vl);
    vuint32m4_t v_shift_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_shift_amt);

    // Perform shifts and merge results based on direction mask
    vuint32m4_t v_aligned_left = __riscv_vsll_vv_u32m4(v_scaled, v_shift_u32, vl);
    vuint32m4_t v_aligned_right = __riscv_vsrl_vv_u32m4(
        v_scaled, __riscv_vneg_v_u32m4(v_shift_u32, vl), vl);
    vuint32m4_t v_aligned = __riscv_vmerge_vvm_u32m4(
        v_aligned_right, v_aligned_left, v_shift_left_mask, vl);

    // Extract fractional part by keeping bottom 16 bits
    vuint32m4_t v_frac = __riscv_vand_vx_u32m4(v_aligned, 0xFFFF, vl);

    // Calculate base segment for LUT lookup
    vuint32m4_t v_base_seg = __riscv_vsrl_vx_u32m4(v_frac, 9, vl);
    vuint16m2_t v_base_seg_u16 = __riscv_vncvt_x_x_w_u16m2(v_base_seg, vl);

    // Calculate offsets for c0 and c1 coefficients
    vuint16m2_t v_offset_c0 = __riscv_vsll_vx_u16m2(v_base_seg_u16, 1, vl);
    vuint16m2_t v_offset_c1 = __riscv_vadd_vx_u16m2(v_offset_c0, 2, vl);

    // Switch to 16-bit element width for gather load to ensure correct data width
    vl = __riscv_vsetvl_e16m2(vl);
    vuint16m2_t v_c0_u16 = __riscv_vluxei16_v_u16m2(kLogLut, v_offset_c0, vl);
    vuint16m2_t v_c1_u16 = __riscv_vluxei16_v_u16m2(kLogLut, v_offset_c1, vl);

    // Switch back to 32-bit element width for remaining computation
    vl = __riscv_vsetvl_e32m4(vl);

    // Widen loaded 16-bit coefficients to 32-bit
    vuint32m4_t v_c0 = __riscv_vwaddu_vx_u32m4(v_c0_u16, 0, vl);
    vuint32m4_t v_c1 = __riscv_vwaddu_vx_u32m4(v_c1_u16, 0, vl);

    // Calculate linear interpolation
    vuint32m4_t v_seg_base = __riscv_vand_vx_u32m4(v_frac, 0xFE00, vl);
    vuint32m4_t v_dist = __riscv_vsub_vv_u32m4(v_frac, v_seg_base, vl);

    // Compute difference between coefficients and multiply by distance
    vint32m4_t v_diff = __riscv_vsub_vv_i32m4(
        __riscv_vreinterpret_v_u32m4_i32m4(v_c1),
        __riscv_vreinterpret_v_u32m4_i32m4(v_c0), vl);
    vint32m4_t v_rel_pos = __riscv_vmul_vv_i32m4(
        v_diff, __riscv_vreinterpret_v_u32m4_i32m4(v_dist), vl);
    v_rel_pos = __riscv_vsra_vx_i32m4(v_rel_pos, kLogScaleLog2, vl);

    // Add interpolation result to base coefficient and fraction
    vuint32m4_t v_final_frac = __riscv_vadd_vv_u32m4(v_frac, v_c0, vl);
    v_final_frac = __riscv_vadd_vv_u32m4(
        v_final_frac, __riscv_vreinterpret_v_i32m4_u32m4(v_rel_pos), vl);

    // Construct final log2 value
    vuint32m4_t v_log2 = __riscv_vsll_vx_u32m4(v_integer, 16, vl);
    v_log2 = __riscv_vadd_vv_u32m4(v_log2, v_final_frac, vl);

    // Convert Log2 to LogE using fixed point multiplication
    vuint32m4_t v_loge = MulHighFixedPoint16_UU_vx_u32m4(v_log2, kLogCoeff, vl);

    // Apply output scaling
    vint32m4_t v_loge_scaled = MulHighFixedPoint16_SU_vx_i32m4(v_loge, output_scale, vl);

    // Saturate result to 16-bit max value
    vint32m4_t v_sat_val = __riscv_vmv_v_x_i32m4(INT16_MAX, vl);
    vint32m4_t v_result = __riscv_vmin_vv_i32m4(v_loge_scaled, v_sat_val, vl);

    // Zero out inactive elements where input was less than or equal to 1
    vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
    v_result = __riscv_vmerge_vvm_i32m4(v_zero, v_result, v_active, vl);

    // Narrow 32-bit result to 16-bit and store
    vint16m2_t v_res_i16 = __riscv_vncvt_x_x_w_i16m2(v_result, vl);
    __riscv_vse16_v_i16m2(output + i, v_res_i16, vl);

    i += vl;
  }
}