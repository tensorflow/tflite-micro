#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"

constexpr uint16_t kLogCoeff = 45426;

const uint16_t kLogLut[] =
{
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
    1094, 963,  830,  695,  559,  421,  282,  142,  0,    0
};

// Calculate Integer Log2 using binary search (SIMD compatible).
// This manual implementation is required because the target architecture 
// (rv32imc_zve32x_zvl128b) does not support the 'zvbb' extension 
// which provides the hardware '__riscv_vclz' instruction.
inline vuint32m4_t VectorLog2Int_Zve32x(vuint32m4_t v_in, size_t vl)
{
  // Initialize variables
  vuint32m4_t v_result = __riscv_vmv_v_x_u32m4(0, vl);
  vuint32m4_t v_tmp;
  vbool8_t v_mask;

  // Check bit 16 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 16, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_mu(v_mask, v_result, v_result, 16, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 8 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 8, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_mu(v_mask, v_result, v_result, 8, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 4 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 4, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_mu(v_mask, v_result, v_result, 4, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 2 and update result and input
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 2, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_mu(v_mask, v_result, v_result, 2, vl);
  v_in = __riscv_vmerge_vvm_u32m4(v_in, v_tmp, v_mask, vl);

  // Check bit 1 and update result
  v_tmp = __riscv_vsrl_vx_u32m4(v_in, 1, vl);
  v_mask = __riscv_vmsne_vx_u32m4_b8(v_tmp, 0, vl);
  v_result = __riscv_vadd_vx_u32m4_mu(v_mask, v_result, v_result, 1, vl);

  return v_result;
}

void FilterbankLogRVV(const uint32_t* input, int num_channels,
                      int32_t output_scale, uint32_t correction_bits,
                      int16_t* output)
{
  const uint32_t* p_src = input;
  int16_t* p_dst = output;
  int remaining = num_channels;

  while (remaining > 0)
  {
    // Set vector length and load input
    size_t vl = __riscv_vsetvl_e32m4(remaining);
    vuint32m4_t v_input = __riscv_vle32_v_u32m4(p_src, vl);
    vuint32m4_t v_scaled = __riscv_vsll_vx_u32m4(v_input, correction_bits, vl);
    vbool8_t v_active = __riscv_vmsgtu_vx_u32m4_b8(v_scaled, 1, vl);

    // Calculate integer part of log2
    vuint32m4_t v_integer = VectorLog2Int_Zve32x(v_scaled, vl);

    // Normalize mantissa to [1.0, 2.0) in Q16
    vuint32m4_t v_shift_norm = __riscv_vrsub_vx_u32m4(v_integer, 31, vl);
    vuint32m4_t v_norm = __riscv_vsll_vv_u32m4(v_scaled, v_shift_norm, vl);
    vuint32m4_t v_frac = __riscv_vsrl_vx_u32m4(v_norm, 15, vl);
    v_frac = __riscv_vand_vx_u32m4(v_frac, 0xFFFF, vl);

    // Calculate base segment index and offsets for LUT access
    vuint32m4_t v_base_seg = __riscv_vsrl_vx_u32m4(v_frac, 9, vl);
    vuint16m2_t v_base_seg_u16 = __riscv_vncvt_x_x_w_u16m2(v_base_seg, vl);
    vuint16m2_t v_offset = __riscv_vsll_vx_u16m2(v_base_seg_u16, 1, vl);

    // Gather LUT coefficients using 16-bit element width
    size_t vl_u16 = __riscv_vsetvl_e16m2(vl);
    vuint16m2_t v_c0_u16 = __riscv_vluxei16_v_u16m2(kLogLut, v_offset, vl_u16);
    v_offset = __riscv_vadd_vx_u16m2(v_offset, 2, vl);
    vuint16m2_t v_c1_u16 = __riscv_vluxei16_v_u16m2(kLogLut, v_offset, vl_u16);

    // Calculate interpolation distance and difference
    vint16m2_t v_diff = __riscv_vsub_vv_i16m2(
        __riscv_vreinterpret_v_u16m2_i16m2(v_c1_u16),
        __riscv_vreinterpret_v_u16m2_i16m2(v_c0_u16), vl_u16);
    vuint16m2_t v_frac_u16 = __riscv_vncvt_x_x_w_u16m2(v_frac, vl);
    vuint16m2_t v_seg_base = __riscv_vand_vx_u16m2(v_frac_u16, 0xFE00, vl_u16);
    vuint16m2_t v_dist = __riscv_vsub_vv_u16m2(v_frac_u16, v_seg_base, vl_u16);

    // Restore vector length and widen for interpolation
    vl = __riscv_vsetvl_e32m4(vl);
    vint32m4_t v_rel_pos = __riscv_vwmul_vv_i32m4(
        v_diff, __riscv_vreinterpret_v_u16m2_i16m2(v_dist), vl);
    v_rel_pos = __riscv_vsra_vx_i32m4(v_rel_pos, 16, vl);

    // Combine interpolated result with base coefficient and fraction
    vint32m4_t v_tmp = __riscv_vwadd_wv_i32m4(
        v_rel_pos, __riscv_vreinterpret_v_u16m2_i16m2(v_c0_u16), vl);
    vint32m4_t v_final_frac_part = __riscv_vadd_vv_i32m4(
        v_tmp, __riscv_vreinterpret_v_u32m4_i32m4(v_frac), vl);

    // Convert Log2 to LogE using fixed point multiplication
    vuint32m4_t v_term1 = __riscv_vmul_vx_u32m4(v_integer, kLogCoeff, vl);
    vuint32m4_t v_frac_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_final_frac_part);
    vuint32m4_t v_term2_u = __riscv_vmul_vx_u32m4(v_frac_u32, kLogCoeff, vl);
    v_term2_u = __riscv_vadd_vx_u32m4(v_term2_u, 32768, vl);
    v_term2_u = __riscv_vsrl_vx_u32m4(v_term2_u, 16, vl);
    vuint32m4_t v_loge = __riscv_vadd_vv_u32m4(v_term1, v_term2_u, vl);

    // Apply output scaling using signed arithmetic
    vint32m4_t v_loge_i = __riscv_vreinterpret_v_u32m4_i32m4(v_loge);
    vint32m4_t v_lo = __riscv_vmul_vx_i32m4(v_loge_i, output_scale, vl);
    vint32m4_t v_hi = __riscv_vmulh_vx_i32m4(v_loge_i, output_scale, vl);

    // Add rounding constant and propagate carry
    vint32m4_t v_lo_rounded = __riscv_vadd_vx_i32m4(v_lo, 32768, vl);
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(
        __riscv_vreinterpret_v_i32m4_u32m4(v_lo_rounded), 32768, vl);
    v_hi = __riscv_vadd_vx_i32m4_mu(v_carry, v_hi, v_hi, 1, vl);

    // Combine high shifted left and low shifted right
    vint32m4_t v_res = __riscv_vor_vv_i32m4(
        __riscv_vsll_vx_i32m4(v_hi, 16, vl),
        __riscv_vreinterpret_v_u32m4_i32m4(
            __riscv_vsrl_vx_u32m4(
                __riscv_vreinterpret_v_i32m4_u32m4(v_lo_rounded), 16, vl)),
        vl);

    // Saturate result to 16-bit range
    vint16m2_t v_res_i16 = __riscv_vnclip_wx_i16m2(v_res, 0, __RISCV_VXRM_RNU, vl);

    // Zero out inactive elements and store result
    vint16m2_t v_zero = __riscv_vmv_v_x_i16m2(0, vl);
    vint16m2_t v_final = __riscv_vmerge_vvm_i16m2(v_zero, v_res_i16, v_active, vl);
    __riscv_vse16_v_i16m2(p_dst, v_final, vl);

    p_src += vl;
    p_dst += vl;
    remaining -= vl;
  }
}