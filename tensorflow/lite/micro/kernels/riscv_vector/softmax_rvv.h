#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SOFTMAX_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SOFTMAX_RVV_H_

#include <riscv_vector.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_log.h"

inline vint32m4_t SaturatingLeftShift_vx_i32m4(vint32m4_t v_in, int shift,
                                               size_t vl)
{
    // Return early if shift is zero or negative
    if (shift <= 0) return v_in;

    // Handle extreme shifts that always saturate
    if (shift >= 31)
    {
        // Create mask for negative values
        vbool8_t v_neg = __riscv_vmslt_vx_i32m4_b8(v_in, 0, vl);
        
        // Set positive max and merge with negative min
        vint32m4_t v_max = __riscv_vmv_v_x_i32m4(INT32_MAX, vl);
        return __riscv_vmerge_vxm_i32m4(v_max, INT32_MIN, v_neg, vl);
    }

    // Perform the logical left shift
    vint32m4_t v_shifted = __riscv_vsll_vx_i32m4(v_in, shift, vl);

    // Verify overflow by shifting back and comparing
    vint32m4_t v_unshifted = __riscv_vsra_vx_i32m4(v_shifted, shift, vl);
    vbool8_t v_no_overflow = __riscv_vmseq_vv_i32m4_b8(v_in, v_unshifted, vl);

    // Select saturating constants based on sign
    vbool8_t v_neg = __riscv_vmslt_vx_i32m4_b8(v_in, 0, vl);
    vint32m4_t v_sat = __riscv_vmerge_vxm_i32m4(
        __riscv_vmv_v_x_i32m4(INT32_MAX, vl), INT32_MIN, v_neg, vl);

    // Merge valid results with saturated results
    return __riscv_vmerge_vvm_i32m4(v_sat, v_shifted, v_no_overflow, vl);
}

inline vint32m4_t MultiplyByQuantizedMultiplierGreaterThanOne_32bit_vx_i32m4(
    vint32m4_t v_x, int32_t multiplier, int left_shift, size_t vl)
{
    // Calculate low 32 bits of product
    vint32m4_t v_lo = __riscv_vmul_vx_i32m4(v_x, multiplier, vl);

    // Calculate high 32 bits of product
    vint32m4_t v_hi = __riscv_vmulh_vx_i32m4(v_x, multiplier, vl);

    // Determine effective right shift amount
    int total_right_shift = 31 - left_shift;

    // Calculate rounding nudge
    int32_t nudge = 1 << (total_right_shift - 1);

    // Add nudge to low part treating as unsigned
    vuint32m4_t v_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_lo);
    vuint32m4_t v_lo_plus_nudge = __riscv_vadd_vx_u32m4(v_lo_u, nudge, vl);

    // Detect carry from low part addition
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_lo_plus_nudge, nudge, vl);

    // Apply carry to high part
    vint32m4_t v_hi_rounded = __riscv_vadd_vx_i32m4_m(v_carry, v_hi, 1, vl);

    // Calculate shift amounts for recombination
    int shift_hi = left_shift + 1;
    int shift_lo = total_right_shift;

    // Shift high part (handling mod 32 behavior)
    vint32m4_t v_res_from_hi;
    if (shift_hi < 32)
    {
        v_res_from_hi = __riscv_vsll_vx_i32m4(v_hi_rounded, shift_hi, vl);
    }
    else
    {
        v_res_from_hi = __riscv_vmv_v_x_i32m4(0, vl);
    }

    // Shift low part
    vuint32m4_t v_res_from_lo =
        __riscv_vsrl_vx_u32m4(v_lo_plus_nudge, shift_lo, vl);

    // Combine results
    return __riscv_vor_vv_i32m4(
        v_res_from_hi, __riscv_vreinterpret_v_u32m4_i32m4(v_res_from_lo), vl);
}

inline vint32m4_t SRMPOT_vx_i32m4(vint32m4_t v_vec, int shift, size_t vl)
{
    // Return early if shift is zero
    if (shift == 0) return v_vec;

    // Handle positive shifts using saturating left shift
    if (shift > 0)
    {
        return SaturatingLeftShift_vx_i32m4(v_vec, shift, vl);
    }
    else
    {
        // Perform rounding arithmetic right shift
        return __riscv_vssra_vx_i32m4(v_vec, -shift, __RISCV_VXRM_RNU, vl);
    }
}

vint32m4_t vectorized_exp_on_negative_values(vint32m4_t v_a_q5_26, size_t vl)
{
    // Define fixed-point constants
    const int kInputFractionalBits = 26;
    const int kOutputFractionalBits = 31;
    const int32_t s_kOneQuarter_q5_26 = INT32_C(1)
                                        << (kInputFractionalBits - 2);
    const int32_t s_mask_val = s_kOneQuarter_q5_26 - 1;

    // Define Taylor Series Constants (Q0.31)
    const int32_t s_result_one_q0_31 = INT32_MAX;
    const int32_t s_exp_neg_1_8_q0_31 = 1895147668;
    const int32_t s_one_third_q0_31 = 715827883;
    const int32_t s_one_24th_q0_31 = 89478485;
    const int32_t s_one_eighth_q0_31 = INT32_C(1)
                                       << (kOutputFractionalBits - 3);

    // Perform range reduction masking
    vint32m4_t v_a_masked = __riscv_vand_vx_i32m4(v_a_q5_26, s_mask_val, vl);

    // Subtract quarter constant
    vint32m4_t v_a_mod_q_m_q_q5_26 =
        __riscv_vsub_vx_i32m4(v_a_masked, s_kOneQuarter_q5_26, vl);

    // Rescale from Q5.26 to Q0.31
    const int rescale_shift = kOutputFractionalBits - kInputFractionalBits;
    vint32m4_t v_a_input_taylor_q0_31 =
        SRMPOT_vx_i32m4(v_a_mod_q_m_q_q5_26, rescale_shift, vl);

    // Center input around -1/8
    vint32m4_t v_y =
        __riscv_vadd_vx_i32m4(v_a_input_taylor_q0_31, s_one_eighth_q0_31, vl);

    // Calculate polynomial terms using 32-bit saturating multiply
    vint32m4_t v_y2 = __riscv_vsmul_vv_i32m4(v_y, v_y, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_y3 = __riscv_vsmul_vv_i32m4(v_y2, v_y, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_y4 = __riscv_vsmul_vv_i32m4(v_y2, v_y2, __RISCV_VXRM_RNU, vl);

    // Calculate coefficients
    vint32m4_t v_term_y2_over_2 = SRMPOT_vx_i32m4(v_y2, -1, vl);
    vint32m4_t v_term_y3_over_3 =
        __riscv_vsmul_vx_i32m4(v_y3, s_one_third_q0_31, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_term_y3_over_6 = SRMPOT_vx_i32m4(v_term_y3_over_3, -1, vl);
    vint32m4_t v_term_y4_over_24 =
        __riscv_vsmul_vx_i32m4(v_y4, s_one_24th_q0_31, __RISCV_VXRM_RNU, vl);

    // Sum polynomial terms
    vint32m4_t v_poly_sum = __riscv_vadd_vv_i32m4(v_y, v_term_y2_over_2, vl);
    v_poly_sum = __riscv_vadd_vv_i32m4(v_poly_sum, v_term_y3_over_6, vl);
    v_poly_sum = __riscv_vadd_vv_i32m4(v_poly_sum, v_term_y4_over_24, vl);

    // Apply constant term
    vint32m4_t v_mul_term = __riscv_vsmul_vx_i32m4(
        v_poly_sum, s_exp_neg_1_8_q0_31, __RISCV_VXRM_RNU, vl);
    vint32m4_t v_current_result =
        __riscv_vadd_vx_i32m4(v_mul_term, s_exp_neg_1_8_q0_31, vl);

    // Calculate remainder for barrel shifter
    vint32m4_t v_remainder_q5_26 =
        __riscv_vsub_vv_i32m4(v_a_mod_q_m_q_q5_26, v_a_q5_26, vl);

    // Multipliers for reconstruction
    const int32_t multipliers[] = {1672461947, 1302514674, 790015084, 290630308,
                                   39332535,   720401,     242};

    // Apply barrel shifter using unrolled loop
    for (int i = 0; i < 7; ++i)
    {
        int exponent = i - 2;
        int shift_amount = 26 + exponent;
        if (shift_amount >= 0 && shift_amount < 32)
        {
            int32_t mask = 1 << shift_amount;
            int32_t mult = multipliers[i];

            vint32m4_t v_rem_masked =
                __riscv_vand_vx_i32m4(v_remainder_q5_26, mask, vl);
            vbool8_t v_apply = __riscv_vmsne_vx_i32m4_b8(v_rem_masked, 0, vl);

            vint32m4_t v_multiplied = __riscv_vsmul_vx_i32m4(
                v_current_result, mult, __RISCV_VXRM_RNU, vl);
            v_current_result = __riscv_vmerge_vvm_i32m4(
                v_current_result, v_multiplied, v_apply, vl);
        }
    }

    // Handle zero input case
    vbool8_t v_zero_mask = __riscv_vmseq_vx_i32m4_b8(v_a_q5_26, 0, vl);
    return __riscv_vmerge_vxm_i32m4(v_current_result, s_result_one_q0_31,
                                    v_zero_mask, vl);
}

template <typename InputT, typename OutputT>
void SoftmaxRVV(const tflite::SoftmaxParams& params,
                const tflite::RuntimeShape& input_shape,
                const InputT* input_data,
                const tflite::RuntimeShape& output_shape, OutputT* output_data)
{
    // Extract quantization parameters
    const int32_t input_beta_multiplier = params.input_multiplier;
    const int32_t input_beta_left_shift = params.input_left_shift;
    const int diff_min = params.diff_min;

    // Define fixed-point constants
    static const int kAccumulationIntegerBits = 12;
    static const int kAccumulationFractionalBits =
        32 - 1 - kAccumulationIntegerBits;
    static const int kExpOutputFractionalBits = 31;

    // Extract shape dimensions
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size = tflite::MatchingFlatSizeSkipDim(
        input_shape, trailing_dim, output_shape);
    const int depth = tflite::MatchingDim(input_shape, trailing_dim,
                                          output_shape, trailing_dim);
    const size_t depth_sz = static_cast<size_t>(depth);

    // Loop over outer dimensions
    for (int i = 0; i < outer_size; ++i)
    {
        const InputT* current_input_data = input_data + i * depth;
        OutputT* current_output_data = output_data + i * depth;

        // Find maximum value in the row
        InputT max_in_row = std::numeric_limits<InputT>::min();
        const InputT* ptr_max = current_input_data;
        size_t n = depth_sz;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e8m1(n);
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_in = __riscv_vle8_v_i8m1(
                    reinterpret_cast<const int8_t*>(ptr_max), vl);
                vint8m1_t v_red = __riscv_vredmax_vs_i8m1_i8m1(
                    v_in, __riscv_vmv_v_x_i8m1(max_in_row, vl), vl);
                max_in_row =
                    std::max(max_in_row, __riscv_vmv_x_s_i8m1_i8(v_red));
            }
            else
            {
                vuint8m1_t v_in = __riscv_vle8_v_u8m1(
                    reinterpret_cast<const uint8_t*>(ptr_max), vl);
                vuint8m1_t v_red = __riscv_vredmaxu_vs_u8m1_u8m1(
                    v_in, __riscv_vmv_v_x_u8m1(max_in_row, vl), vl);
                max_in_row = std::max(max_in_row,
                                      (InputT)__riscv_vmv_x_s_u8m1_u8(v_red));
            }
            ptr_max += vl;
            n -= vl;
        }
        const int32_t max_in_row_s32 = static_cast<int32_t>(max_in_row);

        // Accumulate sum of exponentials
        size_t current_c = 0;
        vint32m1_t v_sum_acc = __riscv_vmv_v_x_i32m1(0, 1);

        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Load and widen input without 64-bit instructions
            vint32m4_t v_input_s32;
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_in = __riscv_vle8_v_i8m1(
                    reinterpret_cast<const int8_t*>(current_input_data +
                                                    current_c),
                    vl);
                vint16m2_t v_in_16 = __riscv_vsext_vf2_i16m2(v_in, vl);
                v_input_s32 = __riscv_vsext_vf2_i32m4(v_in_16, vl);
            }
            else
            {
                vuint8m1_t v_in = __riscv_vle8_v_u8m1(
                    reinterpret_cast<const uint8_t*>(current_input_data +
                                                     current_c),
                    vl);
                vuint16m2_t v_in_16 = __riscv_vzext_vf2_u16m2(v_in, vl);
                vuint32m4_t v_in_32 = __riscv_vzext_vf2_u32m4(v_in_16, vl);
                v_input_s32 = __riscv_vreinterpret_v_u32m4_i32m4(v_in_32);
            }

            // Calculate difference from max
            vint32m4_t v_diff =
                __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);
            vbool8_t v_mask = __riscv_vmsge_vx_i32m4_b8(v_diff, diff_min, vl);

            // Scale difference using custom 32-bit implementation
            vint32m4_t v_diff_scaled =
                MultiplyByQuantizedMultiplierGreaterThanOne_32bit_vx_i32m4(
                    v_diff, input_beta_multiplier, input_beta_left_shift, vl);

            // Calculate exponential
            vint32m4_t v_exp = vectorized_exp_on_negative_values(v_diff_scaled, vl);

            // Rescale result
            vint32m4_t v_exp_rescaled = __riscv_vssra_vx_i32m4(
                v_exp, kExpOutputFractionalBits - kAccumulationFractionalBits,
                __RISCV_VXRM_RNU, vl);

            // Merge and accumulate
            vint32m4_t v_add_val = __riscv_vmerge_vvm_i32m4(
                __riscv_vmv_v_x_i32m4(0, vl), v_exp_rescaled, v_mask, vl);
            v_sum_acc =
                __riscv_vredsum_vs_i32m4_i32m1(v_add_val, v_sum_acc, vl);

            current_c += vl;
        }
        int32_t sum_of_exps = __riscv_vmv_x_s_i32m1_i32(v_sum_acc);

        // Calculate reciprocal
        int num_bits_over_unit;
        int32_t reciprocal = tflite::GetReciprocal(
            sum_of_exps, kAccumulationIntegerBits, &num_bits_over_unit);
        const int exponent = num_bits_over_unit + 31 - (sizeof(OutputT) * 8);
        const int32_t output_min =
            static_cast<int32_t>(std::numeric_limits<OutputT>::min());
        const int32_t output_max =
            static_cast<int32_t>(std::numeric_limits<OutputT>::max());

        // Compute final output
        current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Reload and widen input
            vint32m4_t v_input_s32;
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_in = __riscv_vle8_v_i8m1(
                    reinterpret_cast<const int8_t*>(current_input_data + current_c), vl);
                v_input_s32 = __riscv_vsext_vf2_i32m4(
                    __riscv_vsext_vf2_i16m2(v_in, vl), vl);
            }
            else
            {
                vuint8m1_t v_in = __riscv_vle8_v_u8m1(
                    reinterpret_cast<const uint8_t*>(current_input_data + current_c), vl);
                v_input_s32 = __riscv_vreinterpret_v_u32m4_i32m4(
                    __riscv_vzext_vf2_u32m4(__riscv_vzext_vf2_u16m2(v_in, vl), vl));
            }

            // Recompute difference and mask
            vint32m4_t v_diff =
                __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);
            vbool8_t v_mask = __riscv_vmsge_vx_i32m4_b8(v_diff, diff_min, vl);

            // Scale and exponentiate
            vint32m4_t v_diff_scaled =
                MultiplyByQuantizedMultiplierGreaterThanOne_32bit_vx_i32m4(
                    v_diff, input_beta_multiplier, input_beta_left_shift, vl);
            vint32m4_t v_exp = vectorized_exp_on_negative_values(v_diff_scaled, vl);

            // Multiply by reciprocal using 32-bit saturating multiply
            vint32m4_t v_prod = __riscv_vsmul_vx_i32m4(v_exp, reciprocal,
                                                       __RISCV_VXRM_RNU, vl);

            // Perform final shift and add offset
            vint32m4_t v_out_shifted = __riscv_vssra_vx_i32m4(
                v_prod, exponent, __RISCV_VXRM_RNU, vl);
            vint32m4_t v_out_final =
                __riscv_vadd_vx_i32m4(v_out_shifted, output_min, vl);

            // Clamp result
            v_out_final = __riscv_vmax_vx_i32m4(v_out_final, output_min, vl);
            v_out_final = __riscv_vmin_vx_i32m4(v_out_final, output_max, vl);

            // Apply mask using vector merge
            v_out_final = __riscv_vmerge_vvm_i32m4(
                __riscv_vmv_v_x_i32m4(output_min, vl), v_out_final, v_mask, vl);

            // Narrow and store result
            if constexpr (sizeof(OutputT) == 1)
            {
                if constexpr (std::is_signed_v<OutputT>)
                {
                    vint8m1_t v_store = __riscv_vncvt_x_x_w_i8m1(
                        __riscv_vncvt_x_x_w_i16m2(v_out_final, vl), vl);
                    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(
                                            current_output_data + current_c),
                                        v_store, vl);
                }
                else
                {
                    vuint8m1_t v_store = __riscv_vncvt_x_x_w_u8m1(
                        __riscv_vncvt_x_x_w_u16m2(
                            __riscv_vreinterpret_v_i32m4_u32m4(v_out_final),
                            vl),
                        vl);
                    __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(
                                            current_output_data + current_c),
                                        v_store, vl);
                }
            }
            else
            {
                vint16m2_t v_store = __riscv_vncvt_x_x_w_i16m2(v_out_final, vl);
                __riscv_vse16_v_i16m2(
                    reinterpret_cast<int16_t*>(current_output_data + current_c),
                    v_store, vl);
            }
            current_c += vl;
        }
    }
}

#endif