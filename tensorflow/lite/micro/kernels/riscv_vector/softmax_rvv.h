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

// Vectorized Saturating Rounding Doubling High Multiply (Vector-Vector)
inline vint32m4_t SRDMH_vv_i32m4(vint32m4_t v_a, vint32m4_t v_b, size_t vl)
{
    // Define constants for saturation and rounding nudge
    const int32_t s_int32_min = INT32_MIN;
    const int32_t s_int32_max = INT32_MAX;
    const int32_t s_nudge_pos = (INT32_C(1) << 30); // 05 represented as Q31
    const int32_t s_nudge_neg = 1 - (INT32_C(1) << 30); // -05 represented as Q31

    // Check for overflow condition (INT32_MIN * INT32_MIN)
    vbool8_t v_min_mask_a = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    vbool8_t v_min_mask_b = __riscv_vmseq_vx_i32m4_b8(v_b, s_int32_min, vl);
    vbool8_t v_overflow_mask = __riscv_vmand_mm_b8(v_min_mask_a, v_min_mask_b, vl);

    // Calculate the 64-bit product using 32-bit low and high parts
    vint32m4_t v_prod_lo = __riscv_vmul_vv_i32m4(v_a, v_b, vl);
    vint32m4_t v_prod_hi = __riscv_vmulh_vv_i32m4(v_a, v_b, vl);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);

    // Determine the sign of the product to apply the correct rounding nudge
    vint32m4_t v_xor_signs = __riscv_vxor_vv_i32m4(v_a, v_b, vl);
    vbool8_t v_prod_sign_pos_mask = __riscv_vmsge_vx_i32m4_b8(v_xor_signs, 0, vl);

    // Select the appropriate nudge value (positive or negative 05 in Q31)
    vint32m4_t v_nudge_lo = __riscv_vmv_v_x_i32m4(s_nudge_neg, vl);
    v_nudge_lo = __riscv_vmerge_vxm_i32m4(v_nudge_lo, s_nudge_pos, v_prod_sign_pos_mask, vl);
    vuint32m4_t v_nudge_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_nudge_lo);

    // Prepare the high part of the nudge (-1 for negative, 0 for positive)
    vint32m4_t v_nudge_hi = __riscv_vmv_v_x_i32m4(-1, vl);
    v_nudge_hi = __riscv_vmerge_vxm_i32m4(v_nudge_hi, 0, v_prod_sign_pos_mask, vl);

    // Add the 64-bit nudge value to the 64-bit product using 32-bit vector ops with carry
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vv_u32m4(v_prod_lo_u, v_nudge_lo_u, vl);
    vbool8_t v_carry_mask = __riscv_vmsltu_vv_u32m4_b8(v_sum_lo_u, v_prod_lo_u, vl);
    vint32m4_t v_sum_hi = __riscv_vadd_vv_i32m4(v_prod_hi, v_nudge_hi, vl);
    v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry_mask, v_sum_hi, 1, vl);

    // Perform the effective right shift by 31 (doubling high multiply)
    vuint32m4_t v_sum_lo_shifted_u = __riscv_vsrl_vx_u32m4(v_sum_lo_u, 31, vl);
    vint32m4_t v_sum_hi_shifted = __riscv_vsll_vx_i32m4(v_sum_hi, 1, vl);
    vint32m4_t v_result_before_sat = __riscv_vor_vv_i32m4(v_sum_hi_shifted, __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_shifted_u), vl);

    // Apply saturation for the overflow case (INT32_MIN * INT32_MIN)
    vint32m4_t v_result = __riscv_vmerge_vxm_i32m4(v_result_before_sat, s_int32_max, v_overflow_mask, vl);

    // Return the final saturated and rounded result
    return v_result;
}

// Vectorized Saturating Rounding Doubling High Multiply (Vector-Scalar)
inline vint32m4_t SRDMH_vx_i32m4(vint32m4_t v_a, int32_t s_b, size_t vl)
{
    // Define constants for saturation and rounding nudge
    const int32_t s_int32_min = INT32_MIN;
    const int32_t s_int32_max = INT32_MAX;
    const int32_t s_nudge_pos = (INT32_C(1) << 30); // 05 represented as Q31
    const int32_t s_nudge_neg = 1 - (INT32_C(1) << 30); // -05 represented as Q31

    // Check for overflow condition (vector elements == INT32_MIN and s_b == INT32_MIN)
    vbool8_t v_overflow_mask;
    if (s_b == s_int32_min)
    {
        v_overflow_mask = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    } 
    else
    {
        // Create a mask that is all false if scalar is not INT32_MIN
        vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
        v_overflow_mask = __riscv_vmslt_vv_i32m4_b8(v_zero, v_zero, vl); // Always false
    }

    // Calculate the 64-bit product using 32-bit low and high parts
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_a, s_b, vl);
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_a, s_b, vl);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);

    // Determine the sign of the product to apply the correct rounding nudge
    vint32m4_t v_xor_signs = __riscv_vxor_vx_i32m4(v_a, s_b, vl);
    vbool8_t v_prod_sign_pos_mask = __riscv_vmsge_vx_i32m4_b8(v_xor_signs, 0, vl);

    // Select the appropriate nudge value (positive or negative 05 in Q31)
    vint32m4_t v_nudge_lo = __riscv_vmv_v_x_i32m4(s_nudge_neg, vl);
    v_nudge_lo = __riscv_vmerge_vxm_i32m4(v_nudge_lo, s_nudge_pos, v_prod_sign_pos_mask, vl);
    vuint32m4_t v_nudge_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_nudge_lo);

    // Prepare the high part of the nudge (-1 for negative, 0 for positive)
    vint32m4_t v_nudge_hi = __riscv_vmv_v_x_i32m4(-1, vl);
    v_nudge_hi = __riscv_vmerge_vxm_i32m4(v_nudge_hi, 0, v_prod_sign_pos_mask, vl);

    // Add the 64-bit nudge value to the 64-bit product using 32-bit vector ops with carry
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vv_u32m4(v_prod_lo_u, v_nudge_lo_u, vl);
    vbool8_t v_carry_mask = __riscv_vmsltu_vv_u32m4_b8(v_sum_lo_u, v_prod_lo_u, vl);
    vint32m4_t v_sum_hi = __riscv_vadd_vv_i32m4(v_prod_hi, v_nudge_hi, vl);
    v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry_mask, v_sum_hi, 1, vl);

    // Perform the effective right shift by 31 (doubling high multiply)
    vuint32m4_t v_sum_lo_shifted_u = __riscv_vsrl_vx_u32m4(v_sum_lo_u, 31, vl);
    vint32m4_t v_sum_hi_shifted = __riscv_vsll_vx_i32m4(v_sum_hi, 1, vl);
    vint32m4_t v_result_before_sat = __riscv_vor_vv_i32m4(v_sum_hi_shifted, __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_shifted_u), vl);

    // Apply saturation for the overflow case
    vint32m4_t v_result = __riscv_vmerge_vxm_i32m4(v_result_before_sat, s_int32_max, v_overflow_mask, vl);

    // Return the final saturated and rounded result
    return v_result;
}

// Vectorized Saturating Rounded Multiply by Power of Two (Vector-Scalar)
inline vint32m4_t SRMPOT_vx_i32m4(vint32m4_t v_vec, int shift, size_t vl)
{
    // Handle left shifts (shift > 0)
    if (shift > 0) 
    {
        // Cast shift amount to signed 32-bit integer
        const int32_t s_shift = shift;
        // If shift is zero, return the original vector
        if (s_shift == 0) return v_vec;

        // Define saturation limits
        const int32_t s_max_val = INT32_MAX;
        const int32_t s_min_val = INT32_MIN;

        // Handle large left shifts (>= 31), resulting in saturation or zero
        if (s_shift >= 31) 
        {
             vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
             vbool8_t v_pos_mask = __riscv_vmsgt_vx_i32m4_b8(v_vec, 0, vl);
             vbool8_t v_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, 0, vl);
             // Saturate positive values to max, negative values to min
             vint32m4_t v_saturated = __riscv_vmerge_vxm_i32m4(v_zero, s_max_val, v_pos_mask, vl);
             v_saturated = __riscv_vmerge_vxm_i32m4(v_saturated, s_min_val, v_neg_mask, vl);
             return v_saturated;
        } 
        else 
        {
            // Calculate saturation thresholds for smaller left shifts
            const int32_t scalar_type_bits = 32;
            const int64_t pos_threshold_64 = (INT64_C(1) << (scalar_type_bits - 1 - s_shift)) - 1;
            const int64_t neg_threshold_64 = -(INT64_C(1) << (scalar_type_bits - 1 - s_shift));

            // Clamp thresholds to int32 range
            const int32_t pos_threshold = (pos_threshold_64 > INT32_MAX) ? INT32_MAX : (int32_t)pos_threshold_64;
            const int32_t neg_threshold = (neg_threshold_64 < INT32_MIN) ? INT32_MIN : (int32_t)neg_threshold_64;

            // Create masks for elements exceeding positive or negative thresholds
            vbool8_t v_pos_ovfl_mask = __riscv_vmsgt_vx_i32m4_b8(v_vec, pos_threshold, vl);
            vbool8_t v_neg_ovfl_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, neg_threshold, vl);

            // Perform the left shift
            vint32m4_t v_shifted = __riscv_vsll_vx_i32m4(v_vec, s_shift, vl);

            // Merge shifted results with saturation values based on overflow masks
            vint32m4_t v_result = __riscv_vmerge_vxm_i32m4(v_shifted, s_max_val, v_pos_ovfl_mask, vl);
            v_result = __riscv_vmerge_vxm_i32m4(v_result, s_min_val, v_neg_ovfl_mask, vl);
            return v_result;
        }
    } 
    else if (shift == 0) 
    { // Handle no shift (shift == 0)
         return v_vec;
    } 
    else // Handle right shifts (shift < 0)
    {
        // Calculate the positive exponent for right shift, capped at 31
        int exponent = -shift;
        exponent = std::min(31, exponent);
        // If exponent is zero, return the original vector
        if (exponent == 0) return v_vec;

        // Calculate the mask for extracting the remainder bits
        const int32_t s_mask_val = (INT32_C(1) << exponent) - 1;
        const int32_t s_zero = 0;
        const int32_t s_one = 1;

        // Extract the remainder and perform the arithmetic right shift
        vint32m4_t v_remainder = __riscv_vand_vx_i32m4(v_vec, s_mask_val, vl);
        vint32m4_t v_shifted = __riscv_vsra_vx_i32m4(v_vec, exponent, vl);

        // Calculate the rounding threshold (half the divisor)
        const int32_t s_threshold_base = s_mask_val >> 1;
        vint32m4_t v_threshold = __riscv_vmv_v_x_i32m4(s_threshold_base, vl);
        // Adjust threshold for negative numbers (round away from zero)
        vbool8_t v_is_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, s_zero, vl);
        v_threshold = __riscv_vadd_vx_i32m4_m(v_is_neg_mask, v_threshold, s_one, vl);

        // Create mask for elements where remainder > threshold, requiring rounding up
        vbool8_t v_add1_mask = __riscv_vmsgt_vv_i32m4_b8(v_remainder, v_threshold, vl);
        // Add 1 to the shifted result for elements needing rounding up
        vint32m4_t v_result = __riscv_vadd_vx_i32m4_m(v_add1_mask, v_shifted, s_one, vl);

        // Return the rounded right-shifted result
        return v_result;
    }
}

// Vectorized MultiplyByQuantizedMultiplier for multipliers > 1 (Vector-Scalar)
inline vint32m4_t MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
    vint32m4_t v_x, int32_t quantized_multiplier, int left_shift, size_t vl) 
{

    // Perform saturating left shift using SRMPOT
    vint32m4_t v_shifted_x = SRMPOT_vx_i32m4(v_x, left_shift, vl);

    // Perform saturating rounding doubling high multiply using SRDMH
    return SRDMH_vx_i32m4(v_shifted_x, quantized_multiplier, vl);
}

// Vectorized exp function for quantized negative inputs
vint32m4_t vectorized_exp_on_negative_values(vint32m4_t v_a_q5_26, size_t vl)
{
    // Define quantization parameters and constants
    const int kInputIntegerBits = 5; // Q526 input format
    const int kInputFractionalBits = 32 - 1 - kInputIntegerBits;
    const int kOutputFractionalBits = 31; // Q031 output format

    // Constants for input range reduction (modulo 1/4)
    const int32_t s_kOneQuarter_q5_26 = INT32_C(1) << (kInputFractionalBits - 2);
    const int32_t s_mask_val = s_kOneQuarter_q5_26 - 1;

    // Constants for Taylor series approximation and final result assembly
    const int32_t s_result_one_q0_31 = INT32_MAX; // 10 in Q031
    const int32_t s_exp_neg_1_8_q0_31 = 1895147668; // exp(-1/8) in Q031
    const int32_t s_one_third_q0_31 = 715827883; // 1/3 in Q031
    const int32_t s_one_eighth_q0_31 = INT32_C(1) << (kOutputFractionalBits - 3); // 1/8 in Q031

    // Reduce input `a` to the range [-1/4, 0] by finding `a mod (-1/4)`
    vint32m4_t v_a_masked = __riscv_vand_vx_i32m4(v_a_q5_26, s_mask_val, vl);
    vint32m4_t v_a_mod_q_m_q_q5_26 = __riscv_vsub_vx_i32m4(v_a_masked, s_kOneQuarter_q5_26, vl);
    // Store the multiple of -1/4 that was subtracted
    vint32m4_t v_remainder_q5_26 = __riscv_vsub_vv_i32m4(v_a_q5_26, v_a_mod_q_m_q_q5_26, vl);

    // Rescale the reduced input from Q526 to Q031 for Taylor series input
    const int rescale_shift = kInputIntegerBits - 0;
    vint32m4_t v_a_input_taylor_q0_31 = SRMPOT_vx_i32m4(v_a_mod_q_m_q_q5_26, -rescale_shift, vl);

    // Calculate Taylor series approximation for exp(x) around x = -1/8 Let y = x + 1/8
    vint32m4_t v_y = __riscv_vadd_vx_i32m4(v_a_input_taylor_q0_31, s_one_eighth_q0_31, vl);

    // Calculate powers of y needed for the Taylor expansion (y^2, y^3, y^4)
    vint32m4_t v_y2 = SRDMH_vv_i32m4(v_y, v_y, vl);
    vint32m4_t v_y3 = SRDMH_vv_i32m4(v_y2, v_y, vl);
    vint32m4_t v_y4 = SRDMH_vv_i32m4(v_y2, v_y2, vl);

    // Calculate term y^4 / 4
    vint32m4_t v_y4_over_4 = SRMPOT_vx_i32m4(v_y4, -2, vl);

    // Combine Taylor series terms: exp(-1/8) * (1 + y + y^2/2! + y^3/3! + y^4/4! + )
    // Approximation used: exp(-1/8) * (1 + y + (y^2 + (y^3 + y^4/4) / 3) / 2)
    vint32m4_t v_term1 = __riscv_vadd_vv_i32m4(v_y4_over_4, v_y3, vl);
    vint32m4_t v_term2 = SRDMH_vx_i32m4(v_term1, s_one_third_q0_31, vl);
    vint32m4_t v_term3 = __riscv_vadd_vv_i32m4(v_term2, v_y2, vl);
    vint32m4_t v_sum_of_higher_terms = SRMPOT_vx_i32m4(v_term3, -1, vl); // Division by 2

    // Calculate the term inside the main bracket (y + higher terms)
    vint32m4_t v_bracket_term = __riscv_vadd_vv_i32m4(v_y, v_sum_of_higher_terms, vl);

    // Multiply bracket term by precomputed exp(-1/8)
    vint32m4_t v_const_term_vec = __riscv_vmv_v_x_i32m4(s_exp_neg_1_8_q0_31, vl);
    vint32m4_t v_mul_term = SRDMH_vv_i32m4(v_bracket_term, v_const_term_vec, vl);

    // Add the constant term exp(-1/8) to complete the Taylor approximation for the interval [-1/4, 0]
    vint32m4_t v_interval_result_q0_31 = __riscv_vadd_vv_i32m4(v_mul_term, v_const_term_vec, vl);

    // Start reconstructing the full result using the remainder and precomputed exp factors
    vint32m4_t v_current_result = v_interval_result_q0_31;

    // Define precomputed multipliers exp(-1/4), exp(-1/2), exp(-1), etc in Q031
    const int32_t s_mult_exp_neg_1_4 = 1672461947;
    const int32_t s_mult_exp_neg_1_2 = 1302514674;
    const int32_t s_mult_exp_neg_1   = 790015084;
    const int32_t s_mult_exp_neg_2   = 290630308;
    const int32_t s_mult_exp_neg_4   = 39332535;
    const int32_t s_mult_exp_neg_8   = 720401;
    const int32_t s_mult_exp_neg_16  = 242;

    // Macro to apply barrel shifter logic: multiply by exp(-2^k) if corresponding bit in remainder is set
    #define APPLY_BARREL_SHIFT(exponent, multiplier_q0_31) \
    do { \
        /* Check if the exponent is within the representable input integer bits */ \
        if (kInputIntegerBits > exponent) { \
            /* Calculate the bit position corresponding to this exponent in the Q526 remainder */ \
            const int shift_amount = kInputFractionalBits + exponent; \
            /* Ensure the bit position is valid */ \
            if (shift_amount >= 0 && shift_amount < 32) { \
                /* Create a mask for the specific bit */ \
                int32_t bit_mask_val = INT32_C(1) << shift_amount; \
                /* Check if the bit is set in the remainder vector */ \
                vint32m4_t v_rem_masked = __riscv_vand_vx_i32m4(v_remainder_q5_26, bit_mask_val, vl); \
                vbool8_t v_apply_mask = __riscv_vmsne_vx_i32m4_b8(v_rem_masked, 0, vl); \
                /* Multiply the current result by the precomputed factor */ \
                vint32m4_t v_multiplied = SRDMH_vx_i32m4(v_current_result, multiplier_q0_31, vl); \
                /* Merge the multiplied result where the mask is true */ \
                v_current_result = __riscv_vmerge_vvm_i32m4(v_current_result, v_multiplied, v_apply_mask, vl); \
            } \
        } \
    } while(0)

    // Apply the barrel shifter for each power of 2 from exp(-1/4) up to exp(-16)
    APPLY_BARREL_SHIFT(-2, s_mult_exp_neg_1_4);
    APPLY_BARREL_SHIFT(-1, s_mult_exp_neg_1_2);
    APPLY_BARREL_SHIFT( 0, s_mult_exp_neg_1);
    APPLY_BARREL_SHIFT( 1, s_mult_exp_neg_2);
    APPLY_BARREL_SHIFT( 2, s_mult_exp_neg_4);
    APPLY_BARREL_SHIFT( 3, s_mult_exp_neg_8);
    APPLY_BARREL_SHIFT( 4, s_mult_exp_neg_16);

    // Undefine the helper macro
    #undef APPLY_BARREL_SHIFT

    // Handle the special case where input a = 0, the result should be 10 (INT32_MAX in Q031)
    vint32m4_t v_final_result = v_current_result;
    vbool8_t v_zero_mask = __riscv_vmseq_vx_i32m4_b8(v_a_q5_26, 0, vl);
    v_final_result = __riscv_vmerge_vxm_i32m4(v_final_result, s_result_one_q0_31, v_zero_mask, vl);

    // Return the final computed exp result in Q031 format
    return v_final_result;
}

// Vectorized Softmax implementation for INT8 input and INT8/INT16 output
template<typename OutputT>
void SoftmaxInt8RVV(const tflite::SoftmaxParams& params,
                    const tflite::RuntimeShape& input_shape,
                    const int8_t* input_data,
                    const tflite::RuntimeShape& output_shape,
                    OutputT* output_data)
{
    // Extract Softmax quantization parameters
    const int32_t input_beta_multiplier = params.input_multiplier;
    const int32_t input_beta_left_shift = params.input_left_shift;
    const int diff_min = params.diff_min;

    // Define fixed-point format for intermediate sum accumulation (Q1219)
    static const int kAccumulationIntegerBits = 12;
    static const int kAccumulationFractionalBits = 32 - 1 - kAccumulationIntegerBits;

    // Define fixed-point format for exp function output (Q031)
    static const int kExpOutputFractionalBits = 31;

    // Get input/output shape dimensions
    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size = tflite::MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth = tflite::MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const size_t depth_sz = static_cast<size_t>(depth);

    // Loop over each row (batch or outer dimension)
    for (int i = 0; i < outer_size; ++i)
    {
        // Set input and output pointers for the current row
        const int8_t* current_input_data = input_data + i * depth;
        OutputT* current_output_data = output_data + i * depth;

        // Initialize scalar and vector max accumulators
        int8_t max_in_row = std::numeric_limits<int8_t>::min();
        size_t vl_temp = __riscv_vsetvl_e8m1(1); // Set VL=1 for scalar init
        vint8m1_t v_max_acc_m1 = __riscv_vmv_v_x_i8m1(max_in_row, vl_temp);

        // Process the row in vector chunks to find the maximum value
        const int8_t* Ptr_max = current_input_data;
        for (ptrdiff_t n = depth_sz; n > 0; ) {
            size_t vl = __riscv_vsetvl_e8m1(n);
            vint8m1_t v_input_m1 = __riscv_vle8_v_i8m1(Ptr_max, vl);
            // Perform reduction max across the loaded vector and the accumulator
            v_max_acc_m1 = __riscv_vredmax_vs_i8m1_i8m1(v_input_m1, v_max_acc_m1, vl);
            Ptr_max += vl;
            n -= vl;
        }

        // Extract the final scalar maximum value from the vector accumulator
        max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_acc_m1);
        const int32_t max_in_row_s32 = static_cast<int32_t>(max_in_row);

        // Initialize vector sum accumulator (using m1 for reduction target)
        vl_temp = __riscv_vsetvl_e32m1(1); // Set VL=1 for scalar init
        vint32m1_t v_sum_acc_m1 = __riscv_vmv_v_x_i32m1(0, vl_temp);

        // Process the row in vector chunks
        size_t current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Load input chunk (i8), widen to i16, then widen to i32
            vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(current_input_data + current_c, vl);
            vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
            vint32m4_t v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl);

            // Calculate difference from max (input - max_in_row)
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);

            // Create mask for elements where difference >= diff_min
            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl);

            // Rescale difference for exp input (Q526 format)
            vint32m4_t v_diff_rescaled_q5_26 = MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
                v_diff_s32, input_beta_multiplier, input_beta_left_shift, vl);

            // Calculate exponent using the vectorized exp function (output in Q031)
            vint32m4_t v_exp_val_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled_q5_26, vl);

            // Rescale exponent result from Q031 to accumulation format (Q1219)
            const int rescale_shift_exp_to_accum = kExpOutputFractionalBits - kAccumulationFractionalBits;
            vint32m4_t v_exp_term_q12_19 = SRMPOT_vx_i32m4(v_exp_val_q0_31, -rescale_shift_exp_to_accum, vl);

            // Zero out exponent terms where difference was below diff_min threshold
            vint32m4_t v_zero_q12_19 = __riscv_vmv_v_x_i32m4(0, vl);
            vint32m4_t v_exp_term_masked_q12_19 = __riscv_vmerge_vvm_i32m4(v_zero_q12_19, v_exp_term_q12_19, v_diff_mask, vl);

            // Accumulate the sum using vector reduction sum
            v_sum_acc_m1 = __riscv_vredsum_vs_i32m4_i32m1(v_exp_term_masked_q12_19, v_sum_acc_m1, vl);

            // Advance pointer for the next chunk
            current_c += vl;
        }
        
        // Extract the final scalar sum from the vector accumulator
        int32_t sum_of_exps_raw = __riscv_vmv_x_s_i32m1_i32(v_sum_acc_m1);

        // Calculate the reciprocal of the sum (Q031 format)
        int num_bits_over_unit; // Headroom bits for reciprocal calculation
        int32_t reciprocal_raw_q0_31 = tflite::GetReciprocal(sum_of_exps_raw, kAccumulationIntegerBits, &num_bits_over_unit);
        const int32_t s_shifted_scale_raw_q0_31 = reciprocal_raw_q0_31;

        // Calculate the final right shift amount needed to scale to the output format
        const int output_bits = sizeof(OutputT) * 8;
        const int exponent = num_bits_over_unit + 31 - output_bits;

        // Get output type min/max values for saturation
        const OutputT output_min_val = std::numeric_limits<OutputT>::min();
        const OutputT output_max_val = std::numeric_limits<OutputT>::max();
        const int32_t output_min_s32 = static_cast<int32_t>(output_min_val);
        const int32_t output_max_s32 = static_cast<int32_t>(output_max_val);

        // Process the row again in vector chunks to calculate and store output
        current_c = 0;
        while (current_c < depth_sz)
        {
             size_t vl = __riscv_vsetvl_e32m4(depth_sz - current_c);

            // Reload input chunk and recalculate exp (or store/reload from previous step if memory allows)
            vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(current_input_data + current_c, vl);
            vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
            vint32m4_t v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl);
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl);
            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl);
            vint32m4_t v_diff_rescaled_q5_26 = MultiplyByQuantizedMultiplierGreaterThanOne_vx_i32m4(
                v_diff_s32, input_beta_multiplier, input_beta_left_shift, vl);
            vint32m4_t v_exp_in_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled_q5_26, vl);

            // Multiply the exponent by the reciprocal scale (SRDMH gives Q031 result)
            vint32m4_t v_product_raw_q0_31 = SRDMH_vx_i32m4(v_exp_in_q0_31, s_shifted_scale_raw_q0_31, vl);

            // Apply the final right shift to scale to the output range
            vint32m4_t v_unsat_output = SRMPOT_vx_i32m4(v_product_raw_q0_31, -exponent, vl);

            // Add the output offset (min value of OutputT) Note: TFLM reference uses this instead of output_offset param
            vint32m4_t v_shifted_output = __riscv_vadd_vx_i32m4(v_unsat_output, output_min_s32, vl);

            // Clamp the result to the output type's min/max range
            vint32m4_t v_clamped_output = __riscv_vmax_vx_i32m4(v_shifted_output, output_min_s32, vl);
            v_clamped_output = __riscv_vmin_vx_i32m4(v_clamped_output, output_max_s32, vl);

            // Merge the clamped output with the minimum output value where the difference was below diff_min
            vint32m4_t v_output_min_vec = __riscv_vmv_v_x_i32m4(output_min_s32, vl);
            vint32m4_t v_final_s32 = __riscv_vmerge_vvm_i32m4(v_output_min_vec, v_clamped_output, v_diff_mask, vl);

            // Narrow the final result to the OutputT (int8 or int16) and store it
            if constexpr (sizeof(OutputT) == 1)
            {
                 // Narrow i32m4 -> i16m2 -> i8m1
                 vint16m2_t v_temp_s16 = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl);
                 vint8m1_t v_final_output = __riscv_vncvt_x_x_w_i8m1(v_temp_s16, vl);

                 // Store i8m1 result
                 __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(current_output_data + current_c), v_final_output, vl);
            }
            else
            {
                 // Narrow i32m4 -> i16m2
                 vint16m2_t v_final_output = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl);

                 // Store i16m2 result
                 __riscv_vse16_v_i16m2(reinterpret_cast<int16_t*>(current_output_data + current_c), v_final_output, vl);
            }

            // Advance pointer for the next chunk
            current_c += vl;
        }
    }
}

#endif