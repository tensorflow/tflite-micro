#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SOFTMAX_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_SOFTMAX_RVV_H_

#include <riscv_vector.h>
#include <limits>
#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_log.h"

// Vectorized absolute value for signed 32-bit integers
inline vint32m4_t vabs_i32m4(vint32m4_t v_in, size_t vl)
{
  // Create a mask for elements that are less than zero
  vbool8_t v_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_in, 0, vl);
  
  // Negate the elements that are negative by calculating (0 - v_in)
  vint32m4_t v_negated = __riscv_vrsub_vx_i32m4(v_in, 0, vl);
  
  // Use the mask to merge the original values (where mask is false) with the negated values
  return __riscv_vmerge_vvm_i32m4(v_in, v_negated, v_neg_mask, vl);
}

// Vectorized Saturating Rounding Doubling High Multiply (Vector-Vector)
inline vint32m4_t SRDMH_vv_i32m4(vint32m4_t v_a, vint32m4_t v_b, size_t vl)
{
    // Define scalar constants for saturation and rounding
    const int32_t s_int32_min = INT32_MIN;
    const int32_t s_int32_max = INT32_MAX;
    const int32_t s_rounding_nudge = (INT32_C(1) << 30);

    // Create a mask for the specific overflow case: INT32_MIN * INT32_MIN
    vbool8_t v_min_mask_a = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    vbool8_t v_min_mask_b = __riscv_vmseq_vx_i32m4_b8(v_b, s_int32_min, vl);
    vbool8_t v_overflow_mask = __riscv_vmand_mm_b8(v_min_mask_a, v_min_mask_b, vl);

    // Perform a 32x32 -> 64-bit multiplication, storing high and low parts
    vint32m4_t v_prod_hi = __riscv_vmulh_vv_i32m4(v_a, v_b, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vv_i32m4(v_a, v_b, vl);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);

    // Add the rounding nudge and detect if a carry-out occurred
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, s_rounding_nudge, vl);
    vbool8_t v_carry_mask = __riscv_vmsltu_vv_u32m4_b8(v_sum_lo_u, v_prod_lo_u, vl);
    
    // Add the carry to the high part of the product
    vint32m4_t v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry_mask, v_prod_hi, 1, vl);

    // Combine the high and low parts to form the doubled result and apply saturation
    vint32m4_t v_result_hi_part = __riscv_vsll_vx_i32m4(v_sum_hi, 1, vl);
    vuint32m4_t v_result_lo_part_u = __riscv_vsrl_vx_u32m4(v_sum_lo_u, 31, vl);
    vint32m4_t v_result_before_sat = __riscv_vor_vv_i32m4(
        v_result_hi_part, 
        __riscv_vreinterpret_v_u32m4_i32m4(v_result_lo_part_u), vl);
    
    // Apply saturation for the INT32_MIN * INT32_MIN case
    return __riscv_vmerge_vxm_i32m4(v_result_before_sat, s_int32_max, v_overflow_mask, vl);
}

// Vectorized Saturating Rounding Doubling High Multiply (Vector-Scalar)
inline vint32m4_t SRDMH_vx_i32m4(vint32m4_t v_a, int32_t s_b, size_t vl)
{
    // Define scalar constants for saturation and rounding
    const int32_t s_int32_min = INT32_MIN;
    const int32_t s_int32_max = INT32_MAX;
    const int32_t s_rounding_nudge = (INT32_C(1) << 30);

    // Create a mask for the specific overflow case: v_a[i] == INT32_MIN and s_b == INT32_MIN
    vbool8_t v_overflow_mask;
    if (s_b == s_int32_min)
    {
        v_overflow_mask = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    }
    else
    {
        vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
        v_overflow_mask = __riscv_vmseq_vx_i32m4_b8(v_zero, 1, vl); // Always false
    }

    // Perform a 32x32 -> 64-bit multiplication, storing high and low parts
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_a, s_b, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_a, s_b, vl);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);

    // Add the rounding nudge and detect if a carry-out occurred
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, s_rounding_nudge, vl);
    vbool8_t v_carry_mask = __riscv_vmsltu_vv_u32m4_b8(v_sum_lo_u, v_prod_lo_u, vl);
    
    // Add the carry to the high part of the product
    vint32m4_t v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry_mask, v_prod_hi, 1, vl);

    // Combine the high and low parts to form the doubled result
    vint32m4_t v_result_hi_part = __riscv_vsll_vx_i32m4(v_sum_hi, 1, vl);
    vuint32m4_t v_result_lo_part_u = __riscv_vsrl_vx_u32m4(v_sum_lo_u, 31, vl);
    vint32m4_t v_result_before_sat = __riscv_vor_vv_i32m4(
        v_result_hi_part,
        __riscv_vreinterpret_v_u32m4_i32m4(v_result_lo_part_u), vl);

    // Apply saturation for the INT32_MIN * INT32_MIN case
    return __riscv_vmerge_vxm_i32m4(v_result_before_sat, s_int32_max, v_overflow_mask, vl);
}

// Vectorized Saturating Rounding Multiply by Power-of-Two
inline vint32m4_t SRMPOT_vx_i32m4(vint32m4_t v_vec, int shift, size_t vl)
{
  // If shift is zero, return the original vector
  if (shift == 0)
  {
    return v_vec;
  }

  // This section handles left shifts (positive shift values)
  if (shift > 0)
  {
    // Define scalar constants for saturation and shifting
    const int32_t s_shift = shift;
    const int32_t s_max_val = INT32_MAX;
    const int32_t s_min_val = INT32_MIN;

    // Handle extreme shifts that always result in saturation
    if (s_shift >= 31)
    {
      vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
      vbool8_t v_pos_mask = __riscv_vmsgt_vx_i32m4_b8(v_vec, 0, vl);
      vbool8_t v_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, 0, vl);
      vint32m4_t v_saturated = __riscv_vmerge_vxm_i32m4(v_zero, s_max_val, v_pos_mask, vl);
      return __riscv_vmerge_vxm_i32m4(v_saturated, s_min_val, v_neg_mask, vl);
    }

    // Calculate thresholds for overflow detection
    const int32_t pos_threshold = (INT32_C(1) << (31 - s_shift));
    const int32_t neg_threshold = -pos_threshold;
    
    // Create masks for positive and negative overflow
    vbool8_t v_pos_ovfl_mask = __riscv_vmsgt_vx_i32m4_b8(v_vec, pos_threshold - 1, vl);
    vbool8_t v_neg_ovfl_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, neg_threshold, vl);

    // Perform the left shift
    vint32m4_t v_shifted = __riscv_vsll_vx_i32m4(v_vec, s_shift, vl);

    // Merge the shifted result with saturated values based on overflow masks
    vint32m4_t v_result = __riscv_vmerge_vxm_i32m4(v_shifted, s_max_val, v_pos_ovfl_mask, vl);
    return __riscv_vmerge_vxm_i32m4(v_result, s_min_val, v_neg_ovfl_mask, vl);

  }
  else
  {
    // This section handles right shifts (negative shift values) with rounding
    const int exponent = -shift;
    if (exponent <= 0) return v_vec;
    
    // Handle extreme shifts that result in 0 or -1
    if (exponent > 31)
    {
        vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
        vbool8_t v_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, 0, vl);
        return __riscv_vmerge_vxm_i32m4(v_zero, -1, v_neg_mask, vl);
    }
    
    // Calculate the rounding threshold ("round half away from zero")
    const int32_t s_mask = (INT32_C(1) << exponent) - 1;
    const int32_t s_threshold_base = s_mask >> 1;
    vbool8_t v_is_negative_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, 0, vl);
    vint32m4_t v_threshold = __riscv_vmv_v_x_i32m4(s_threshold_base, vl);
    v_threshold = __riscv_vadd_vx_i32m4_m(v_is_negative_mask, v_threshold, 1, vl);

    // Check if the remainder requires rounding up
    vint32m4_t v_remainder = __riscv_vand_vx_i32m4(v_vec, s_mask, vl);
    vint32m4_t v_abs_remainder = vabs_i32m4(v_remainder, vl);
    vbool8_t v_should_round_mask = __riscv_vmsgt_vv_i32m4_b8(v_abs_remainder, v_threshold, vl);

    // Perform the arithmetic right shift
    vint32m4_t v_shifted = __riscv_vsra_vx_i32m4(v_vec, exponent, vl);
    
    // Add 1 to the result if rounding is needed
    return __riscv_vadd_vx_i32m4_m(v_should_round_mask, v_shifted, 1, vl);
  }
}

// Vectorized MultiplyByQuantizedMultiplier for multipliers > 1 (Vector-Scalar)
inline vint32m4_t MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
    vint32m4_t v_x, int32_t quantized_multiplier, int left_shift, size_t vl) 
{
    // Apply the left shift to the input vector
    vint32m4_t v_shifted_x = __riscv_vsll_vx_i32m4(v_x, left_shift, vl);
    
    // Perform the saturating rounding doubling high multiply
    return SRDMH_vx_i32m4(v_shifted_x, quantized_multiplier, vl);
}

// Vectorized fixed-point implementation of exp(x) for negative q526 inputs
vint32m4_t vectorized_exp_on_negative_values(vint32m4_t v_a_q5_26, size_t vl)
{
    // Define fixed-point constants for input and output formats
    const int kInputIntegerBits = 5;
    const int kInputFractionalBits = 26; 
    const int kOutputFractionalBits = 31;

    // Define constants for range reduction (exp(x) = exp(x/4) * exp(3x/4))
    const int32_t s_kOneQuarter_q5_26 = INT32_C(1) << (kInputFractionalBits - 2);
    const int32_t s_mask_val = s_kOneQuarter_q5_26 - 1;

    // Define constants for Taylor series approximation of exp(x) around -1/8
    const int32_t s_result_one_q0_31 = INT32_MAX; 
    const int32_t s_exp_neg_1_8_q0_31 = 1895147668; 
    const int32_t s_one_third_q0_31 = 715827883; 
    const int32_t s_one_24th_q0_31 = 89478485; 
    const int32_t s_one_eighth_q0_31 = INT32_C(1) << (kOutputFractionalBits - 3);

    // Perform range reduction to map the input to the [-1/4, 0] interval
    vint32m4_t v_a_masked = __riscv_vand_vx_i32m4(v_a_q5_26, s_mask_val, vl);
    vint32m4_t v_a_mod_q_m_q_q5_26 = __riscv_vsub_vx_i32m4(v_a_masked, s_kOneQuarter_q5_26, vl);
    vint32m4_t v_remainder_q5_26 = __riscv_vsub_vv_i32m4(v_a_mod_q_m_q_q5_26, v_a_q5_26, vl);

    // Rescale for Taylor series input
    const int rescale_shift = kOutputFractionalBits - kInputFractionalBits;
    vint32m4_t v_a_input_taylor_q0_31 = SRMPOT_vx_i32m4(v_a_mod_q_m_q_q5_26, rescale_shift, vl);

    // Center the input around -1/8 for better Taylor series accuracy
    vint32m4_t v_y = __riscv_vadd_vx_i32m4(v_a_input_taylor_q0_31, s_one_eighth_q0_31, vl);

    // Calculate polynomial terms: y^2, y^3, y^4
    vint32m4_t v_y2 = SRDMH_vv_i32m4(v_y, v_y, vl);
    vint32m4_t v_y3 = SRDMH_vv_i32m4(v_y2, v_y, vl);
    vint32m4_t v_y4 = SRDMH_vv_i32m4(v_y2, v_y2, vl);

    // Calculate scaled polynomial terms: y^2/2, y^3/6, y^4/24
    vint32m4_t v_term_y2_over_2 = SRMPOT_vx_i32m4(v_y2, -1, vl);
    vint32m4_t v_term_y3_over_3 = SRDMH_vx_i32m4(v_y3, s_one_third_q0_31, vl);
    vint32m4_t v_term_y3_over_6 = SRMPOT_vx_i32m4(v_term_y3_over_3, -1, vl);
    vint32m4_t v_term_y4_over_24 = SRDMH_vx_i32m4(v_y4, s_one_24th_q0_31, vl);

    // Sum the polynomial terms: y + y^2/2 + y^3/6 + y^4/24
    vint32m4_t v_poly_sum = __riscv_vadd_vv_i32m4(v_y, v_term_y2_over_2, vl);
    v_poly_sum = __riscv_vadd_vv_i32m4(v_poly_sum, v_term_y3_over_6, vl);
    v_poly_sum = __riscv_vadd_vv_i32m4(v_poly_sum, v_term_y4_over_24, vl);
    
    // Calculate the final result for the interval: exp(-1/8) * (1 + poly_sum)
    vint32m4_t v_const_term_vec = __riscv_vmv_v_x_i32m4(s_exp_neg_1_8_q0_31, vl);
    vint32m4_t v_mul_term = SRDMH_vv_i32m4(v_poly_sum, v_const_term_vec, vl);
    vint32m4_t v_interval_result_q0_31 = __riscv_vadd_vv_i32m4(v_mul_term, v_const_term_vec, vl);

    // Reconstruct the full result using a barrel shifter based on the remainder
    vint32m4_t v_current_result = v_interval_result_q0_31;
    const int32_t s_mult_exp_neg_1_4 = 1672461947;
    const int32_t s_mult_exp_neg_1_2 = 1302514674;
    const int32_t s_mult_exp_neg_1   = 790015084;
    const int32_t s_mult_exp_neg_2   = 290630308;
    const int32_t s_mult_exp_neg_4   = 39332535;
    const int32_t s_mult_exp_neg_8   = 720401;
    const int32_t s_mult_exp_neg_16  = 242;

    // Macro to conditionally apply multipliers based on remainder bits
    #define APPLY_BARREL_SHIFT(exponent, multiplier_q0_31) \
    do \
    { \
        if (kInputIntegerBits > exponent) \
        { \
            const int shift_amount = kInputFractionalBits + exponent; \
            if (shift_amount >= 0 && shift_amount < 32) \
            { \
                int32_t bit_mask_val = INT32_C(1) << shift_amount; \
                vint32m4_t v_rem_masked = __riscv_vand_vx_i32m4(v_remainder_q5_26, bit_mask_val, vl); \
                vbool8_t v_apply_mask = __riscv_vmsne_vx_i32m4_b8(v_rem_masked, 0, vl); \
                vint32m4_t v_multiplied = SRDMH_vx_i32m4(v_current_result, multiplier_q0_31, vl); \
                v_current_result = __riscv_vmerge_vvm_i32m4(v_current_result, v_multiplied, v_apply_mask, vl); \
            } \
        } \
    } while(0)

    // Apply barrel shifter for each power-of-two component
    APPLY_BARREL_SHIFT(-2, s_mult_exp_neg_1_4);
    APPLY_BARREL_SHIFT(-1, s_mult_exp_neg_1_2);
    APPLY_BARREL_SHIFT( 0, s_mult_exp_neg_1);
    APPLY_BARREL_SHIFT( 1, s_mult_exp_neg_2);
    APPLY_BARREL_SHIFT( 2, s_mult_exp_neg_4);
    APPLY_BARREL_SHIFT( 3, s_mult_exp_neg_8);
    APPLY_BARREL_SHIFT( 4, s_mult_exp_neg_16);

    #undef APPLY_BARREL_SHIFT

    // Handle the case where input is 0, for which exp(0) = 1
    vint32m4_t v_final_result = v_current_result;
    vbool8_t v_zero_mask = __riscv_vmseq_vx_i32m4_b8(v_a_q5_26, 0, vl);
    v_final_result = __riscv_vmerge_vxm_i32m4(v_final_result, s_result_one_q0_31, v_zero_mask, vl);

    return v_final_result;
}

// Main RVV-accelerated Softmax kernel function
template<typename InputT, typename OutputT>
void SoftmaxRVV(const tflite::SoftmaxParams& params,
                const tflite::RuntimeShape& input_shape,
                const InputT* input_data,
                const tflite::RuntimeShape& output_shape,
                OutputT* output_data)
{
    // Extract quantization parameters
    const int32_t input_beta_multiplier = params.input_multiplier;
    const int32_t input_beta_left_shift = params.input_left_shift;
    const int diff_min = params.diff_min;
    
    // Define fixed-point constants for accumulation and output
    static const int kAccumulationIntegerBits = 12;
    static const int kAccumulationFractionalBits = 32 - 1 - kAccumulationIntegerBits;
    static const int kExpOutputFractionalBits = 31;
    
    // Extract shape dimensions
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size = tflite::MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth = tflite::MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const size_t depth_sz = static_cast<size_t>(depth);

    // Loop over each row in the outer dimensions
    for (int i = 0; i < outer_size; ++i)
    {
        const InputT* current_input_data = input_data + i * depth;
        OutputT* current_output_data = output_data + i * depth;

        // Find the maximum value in the current row for numerical stability
        InputT max_in_row = std::numeric_limits<InputT>::min();
        const InputT* ptr_max = current_input_data;
        ptrdiff_t n = depth_sz;
        while (n > 0)
        {
            size_t vl = __riscv_vsetvl_e8m1(n);
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_input = __riscv_vle8_v_i8m1(reinterpret_cast<const int8_t*>(ptr_max), vl);
                vint8m1_t v_scalar = __riscv_vmv_v_x_i8m1(max_in_row, vl);
                vint8m1_t v_red = __riscv_vredmax_vs_i8m1_i8m1(v_input, v_scalar, vl);
                max_in_row = std::max(max_in_row, __riscv_vmv_x_s_i8m1_i8(v_red));
            }
            else
            {
                vuint8m1_t v_input = __riscv_vle8_v_u8m1(reinterpret_cast<const uint8_t*>(ptr_max), vl);
                vuint8m1_t v_scalar = __riscv_vmv_v_x_u8m1(max_in_row, vl);
                vuint8m1_t v_red = __riscv_vredmaxu_vs_u8m1_u8m1(v_input, v_scalar, vl);
                max_in_row = std::max(max_in_row, __riscv_vmv_x_s_u8m1_u8(v_red));
            }
            ptr_max += vl;
            n -= vl;
        }
        const int32_t max_in_row_s32 = static_cast<int32_t>(max_in_row);

        // Calculate the sum of exponentials of (input - max)
        size_t vl_temp_sum = __riscv_vsetvl_e32m1(1);
        vint32m1_t v_sum_acc_m1 = __riscv_vmv_v_x_i32m1(0, vl_temp_sum);
        size_t current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Load 8-bit input data and widen to 32-bit
            vint32m4_t v_input_s32;
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(reinterpret_cast<const int8_t*>(current_input_data + current_c), vl);
                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
                v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl);
            }
            else
            {
                vuint8m1_t v_input_u8 = __riscv_vle8_v_u8m1(reinterpret_cast<const uint8_t*>(current_input_data + current_c), vl);
                vuint16m2_t v_input_u16 = __riscv_vwaddu_vx_u16m2(v_input_u8, 0, vl);
                vuint32m4_t v_input_u32 = __riscv_vwaddu_vx_u32m4(v_input_u16, 0, vl);
                v_input_s32 = __riscv_vreinterpret_v_u32m4_i32m4(v_input_u32);
            }

            // Calculate the difference and create a mask for values >= diff_min
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);
            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl);
            
            // Rescale the difference for the exp function
            vint32m4_t v_diff_rescaled_q5_26 = MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
                v_diff_s32, input_beta_multiplier, input_beta_left_shift, vl);
            
            // Calculate the exponential of the rescaled difference
            vint32m4_t v_exp_val_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled_q5_26, vl);
            
            // Rescale the exponential result to the accumulation format
            const int rescale_shift = kAccumulationFractionalBits - kExpOutputFractionalBits;
            vint32m4_t v_exp_term_q12_19 = SRMPOT_vx_i32m4(v_exp_val_q0_31, rescale_shift, vl);
            
            // Mask out values that were below the diff_min threshold and accumulate
            vint32m4_t v_zero_q12_19 = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t v_exp_term_masked = __riscv_vmerge_vvm_i32m4(v_zero_q12_19, v_exp_term_q12_19, v_diff_mask, vl);
            v_sum_acc_m1 = __riscv_vredsum_vs_i32m4_i32m1(v_exp_term_masked, v_sum_acc_m1, vl);
            
            current_c += vl;
        }
        int32_t sum_of_exps_raw = __riscv_vmv_x_s_i32m1_i32(v_sum_acc_m1);

        // Calculate the reciprocal of the sum of exponentials
        int num_bits_over_unit;
        int32_t reciprocal_raw_q0_31 = tflite::GetReciprocal(sum_of_exps_raw, kAccumulationIntegerBits, &num_bits_over_unit);
        
        // Calculate the final output shift exponent
        const int exponent = num_bits_over_unit + 31 - (sizeof(OutputT) * 8);
        const int32_t output_min_s32 = static_cast<int32_t>(std::numeric_limits<OutputT>::min());
        const int32_t output_max_s32 = static_cast<int32_t>(std::numeric_limits<OutputT>::max());

        // Compute and store the final output values
        current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Reload and widen the input data
            vint32m4_t v_input_s32;
            if constexpr (std::is_signed_v<InputT>)
            {
                vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(reinterpret_cast<const int8_t*>(current_input_data + current_c), vl);
                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
                v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl);
            }
            else
            {
                vuint8m1_t v_input_u8 = __riscv_vle8_v_u8m1(reinterpret_cast<const uint8_t*>(current_input_data + current_c), vl);
                vuint16m2_t v_input_u16 = __riscv_vwaddu_vx_u16m2(v_input_u8, 0, vl);
                vuint32m4_t v_input_u32 = __riscv_vwaddu_vx_u32m4(v_input_u16, 0, vl);
                v_input_s32 = __riscv_vreinterpret_v_u32m4_i32m4(v_input_u32);
            }
            
            // Recompute the difference, mask, and exponential
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);
            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl);
            vint32m4_t v_diff_rescaled_q5_26 = MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
                v_diff_s32, input_beta_multiplier, input_beta_left_shift, vl);
            vint32m4_t v_exp_in_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled_q5_26, vl);
            
            // Multiply the exponential by the reciprocal to get the normalized result
            vint32m4_t v_product_raw_q0_31 = SRDMH_vx_i32m4(v_exp_in_q0_31, reciprocal_raw_q0_31, vl);
            
            // Rescale the output and add the output offset (zero point)
            vint32m4_t v_unsat_output = SRMPOT_vx_i32m4(v_product_raw_q0_31, -exponent, vl);
            vint32m4_t v_shifted_output = __riscv_vadd_vx_i32m4(v_unsat_output, output_min_s32, vl);
            
            // Clamp the result to the output data type's range
            vint32m4_t v_clamped_output = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(v_shifted_output, output_max_s32, vl), output_min_s32, vl);
            
            // Apply the diff_min mask one last time
            vint32m4_t v_output_min_vec = __riscv_vmv_v_x_i32m4(output_min_s32, vl);
            vint32m4_t v_final_s32 = __riscv_vmerge_vvm_i32m4(v_output_min_vec, v_clamped_output, v_diff_mask, vl);

            // Narrow the 32-bit results down to the output type and store
            if constexpr (sizeof(OutputT) == 1)
            {
                if constexpr (std::is_signed_v<OutputT>)
                {
                    vint16m2_t v_temp_s16 = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl);
                    vint8m1_t v_final_output = __riscv_vncvt_x_x_w_i8m1(v_temp_s16, vl);
                    __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(current_output_data + current_c), v_final_output, vl);
                }
                else
                {
                    vuint32m4_t v_final_u32 = __riscv_vreinterpret_v_i32m4_u32m4(v_final_s32);
                    vuint16m2_t v_temp_u16 = __riscv_vncvt_x_x_w_u16m2(v_final_u32, vl);
                    vuint8m1_t v_final_output = __riscv_vncvt_x_x_w_u8m1(v_temp_u16, vl);
                    __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(current_output_data + current_c), v_final_output, vl);
                }
            }
            else
            {
                vint16m2_t v_final_output = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl);
                __riscv_vse16_v_i16m2(reinterpret_cast<int16_t*>(current_output_data + current_c), v_final_output, vl);
            }
            
            current_c += vl;
        }
    }
}

#endif