#include <riscv_vector.h>
#include <limits>
#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/softmax.h"

constexpr size_t kMaxVLI8M1_H = (128 / 8) * 1;
constexpr size_t kMaxVLI16M2_H = (128 / 16) * 2;
constexpr size_t kMaxVLI32M4_H = (128 / 32) * 4;
constexpr size_t kMaxVL = std::max({ kMaxVLI8M1_H, kMaxVLI16M2_H, kMaxVLI32M4_H });

inline vint32m4_t SRDMH_vv_i32m4(vint32m4_t v_a, vint32m4_t v_b, size_t vl) 
{
    const int32_t s_int32_min = INT32_MIN;
    vbool8_t v_min_mask_a = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    vbool8_t v_min_mask_b = __riscv_vmseq_vx_i32m4_b8(v_b, s_int32_min, vl);
    vbool8_t v_overflow_mask = __riscv_vmand_mm_b8(v_min_mask_a, v_min_mask_b, vl);

    vint32m4_t v_prod_hi = __riscv_vmulh_vv_i32m4(v_a, v_b, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vv_i32m4(v_a, v_b, vl);

    const int32_t s_round_const = (1 << 30);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, s_round_const, vl);
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, s_round_const, vl);

    vint32m4_t v_sum_hi = __riscv_vadd_vx_i32m4(v_prod_hi, 0, vl);
    v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_sum_hi, 1, vl);

    vint32m4_t v_result = v_sum_hi;

    const int32_t s_int32_max = INT32_MAX;
    v_result = __riscv_vmerge_vxm_i32m4(v_result, s_int32_max, v_overflow_mask, vl);

    return v_result;
}

inline vint32m4_t SRDMH_vx_i32m4(vint32m4_t v_a, int32_t s_b, size_t vl) 
{
    const int32_t s_int32_min = INT32_MIN;
    vbool8_t v_overflow_mask;
    if (s_b == s_int32_min) 
    {
         v_overflow_mask = __riscv_vmseq_vx_i32m4_b8(v_a, s_int32_min, vl);
    } else 
    {
        vint32m4_t zero = __riscv_vmv_v_x_i32m4(0, vl);
        v_overflow_mask = __riscv_vmseq_vx_i32m4_b8(zero, 1, vl);
    }

    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_a, s_b, vl);
    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_a, s_b, vl);

    const int32_t s_round_const = (1 << 30);
    vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_prod_lo_u, s_round_const, vl);
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, s_round_const, vl);

    vint32m4_t v_sum_hi = __riscv_vadd_vx_i32m4(v_prod_hi, 0, vl);
    v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_sum_hi, 1, vl);

    vint32m4_t v_result = v_sum_hi;

    const int32_t s_int32_max = INT32_MAX;
    v_result = __riscv_vmerge_vxm_i32m4(v_result, s_int32_max, v_overflow_mask, vl);

    return v_result;
}


inline vint32m4_t SRMPOT_vx_i32m4(vint32m4_t v_vec, int shift, size_t vl) 
{
    if (shift == 0) 
    {
        return v_vec;
    }

    shift = std::max(1, std::min(31, shift));

    const int32_t s_round_mask = (INT64_C(1) << shift) - 1;
    const int32_t s_threshold_base = s_round_mask >> 1;

    vint32m4_t v_remainder = __riscv_vand_vx_i32m4(v_vec, s_round_mask, vl);
    vbool8_t v_is_neg_mask = __riscv_vmslt_vx_i32m4_b8(v_vec, 0, vl);

    vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl);
    vint32m4_t v_neg_adjust = __riscv_vmerge_vxm_i32m4(v_zero, 1, v_is_neg_mask, vl);
    vint32m4_t v_threshold = __riscv_vadd_vx_i32m4(v_neg_adjust, s_threshold_base, vl);

    vbool8_t v_add1_mask = __riscv_vmsgt_vv_i32m4_b8(v_remainder, v_threshold, vl);
    vint32m4_t v_shifted = __riscv_vsra_vx_i32m4(v_vec, shift, vl);
    vint32m4_t v_result = __riscv_vadd_vx_i32m4_m(v_add1_mask, v_shifted, 1, vl);

    return v_result;
}

inline vint32m4_t RoundingMul_vx_i32m4(vint32m4_t v_a, int32_t s_b, size_t vl) 
{
    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_a, s_b, vl);
    vuint32m4_t v_prod_lo = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vmul_vx_i32m4(v_a, s_b, vl));

    const int32_t s_round_offset = (1 << 30);
    vuint32m4_t v_sum_lo = __riscv_vadd_vx_u32m4(v_prod_lo, s_round_offset, vl);
    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo, s_round_offset, vl);

    vint32m4_t v_sum_hi = __riscv_vadd_vx_i32m4(v_prod_hi, 0, vl);
    v_sum_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_sum_hi, 1, vl);

    return v_sum_hi;
}

vint32m4_t vectorized_exp_on_negative_values(vint32m4_t v_a_q5_26, size_t vl) 
{

    const int kInputIntegerBits = 5;
    const int kInputFractionalBits = 32 - 1 - kInputIntegerBits;
    const int kOutputFractionalBits = 31;

    const int32_t s_kOneQuarter_q5_26 = INT32_C(1) << (kInputFractionalBits - 2);
    const int32_t s_mask_val = (INT32_C(1) << (kInputFractionalBits + 2)) - 1;
    const int32_t s_minus_32_q5_26 = INT32_MIN;

    const int32_t s_result_one_q0_31 = INT32_MAX;
    const int32_t s_result_zero_q0_31 = 0;
    const int32_t s_exp_neg_1_8_q0_31 = 1895147668;
    const int32_t s_one_third_q0_31 = 715827883;
    const int32_t s_one_eighth_q0_31 = INT32_C(1) << (kOutputFractionalBits - 3);

    const int32_t s_mult_exp_neg_1_4 = 1672461947;
    const int32_t s_mult_exp_neg_1_2 = 1302514674;
    const int32_t s_mult_exp_neg_1 = 790015084;
    const int32_t s_mult_exp_neg_2 = 290630308;
    const int32_t s_mult_exp_neg_4 = 39332535;
    const int32_t s_mult_exp_neg_8 = 720401;
    const int32_t s_mult_exp_neg_16 = 242;

    vint32m4_t v_a_masked = __riscv_vand_vx_i32m4(v_a_q5_26, s_mask_val, vl);
    vint32m4_t v_a_mod_q_m_q_q5_26 = __riscv_vsub_vx_i32m4(v_a_masked, s_kOneQuarter_q5_26, vl);
    vint32m4_t v_remainder_q5_26 = __riscv_vsub_vv_i32m4(v_a_mod_q_m_q_q5_26, v_a_q5_26, vl);

    const int rescale_shift = kOutputFractionalBits - kInputFractionalBits;
    vint32m4_t v_a_input_taylor_q0_31 = __riscv_vsll_vx_i32m4(v_a_mod_q_m_q_q5_26, rescale_shift, vl);

    vint32m4_t v_x = __riscv_vadd_vx_i32m4(v_a_input_taylor_q0_31, s_one_eighth_q0_31, vl);
    vint32m4_t v_x2 = SRDMH_vv_i32m4(v_x, v_x, vl);
    vint32m4_t v_x3 = SRDMH_vv_i32m4(v_x2, v_x, vl);
    vint32m4_t v_x4 = SRDMH_vv_i32m4(v_x2, v_x2, vl);
    vint32m4_t v_x4_over_4 = SRMPOT_vx_i32m4(v_x4, 2, vl);

    vint32m4_t v_term1 = __riscv_vadd_vv_i32m4(v_x4_over_4, v_x3, vl);
    vint32m4_t v_term2 = SRDMH_vx_i32m4(v_term1, s_one_third_q0_31, vl);
    vint32m4_t v_term3 = __riscv_vadd_vv_i32m4(v_term2, v_x2, vl);
    vint32m4_t v_inner_sum = SRMPOT_vx_i32m4(v_term3, 1, vl);
    vint32m4_t v_bracket_term = __riscv_vadd_vv_i32m4(v_x, v_inner_sum, vl);
    vint32m4_t v_mul_term = SRDMH_vx_i32m4(v_bracket_term, s_exp_neg_1_8_q0_31, vl);
    vint32m4_t v_interval_result_q0_31 = __riscv_vsadd_vx_i32m4(v_mul_term, s_exp_neg_1_8_q0_31, vl);

    vint32m4_t v_current_result = v_interval_result_q0_31;

    #define APPLY_BARREL_SHIFT(exponent, multiplier_q0_31) \
    do { \
        const int shift_amount = kInputFractionalBits + exponent; \
        if (shift_amount >= 0 && shift_amount < 32) { \
            int32_t bit_mask = INT32_C(1) << shift_amount; \
            vbool8_t v_apply_mask = __riscv_vmsne_vx_i32m4_b8( \
                __riscv_vand_vx_i32m4(v_remainder_q5_26, bit_mask, vl), \
                0, vl); \
            vint32m4_t v_multiplied = SRDMH_vx_i32m4(v_current_result, multiplier_q0_31, vl); \
            v_current_result = __riscv_vmerge_vvm_i32m4(v_current_result, v_multiplied, v_apply_mask, vl); \
        } \
    } while(0)

    APPLY_BARREL_SHIFT(-2, s_mult_exp_neg_1_4);
    APPLY_BARREL_SHIFT(-1, s_mult_exp_neg_1_2);
    APPLY_BARREL_SHIFT( 0, s_mult_exp_neg_1);
    APPLY_BARREL_SHIFT( 1, s_mult_exp_neg_2);
    APPLY_BARREL_SHIFT( 2, s_mult_exp_neg_4);
    APPLY_BARREL_SHIFT( 3, s_mult_exp_neg_8);
    APPLY_BARREL_SHIFT( 4, s_mult_exp_neg_16);

    #undef APPLY_BARREL_SHIFT

    vbool8_t v_clamp_mask = __riscv_vmslt_vx_i32m4_b8(v_a_q5_26, s_minus_32_q5_26, vl);
    v_current_result = __riscv_vmerge_vxm_i32m4(v_current_result, s_result_zero_q0_31, v_clamp_mask, vl);

    vbool8_t v_zero_mask = __riscv_vmseq_vx_i32m4_b8(v_a_q5_26, 0, vl);
    v_current_result = __riscv_vmerge_vxm_i32m4(v_current_result, s_result_one_q0_31, v_zero_mask, vl);

    return v_current_result;
}

template<typename OutputT>
void SoftmaxInt8RVV(const tflite::SoftmaxParams& params,
                    const tflite::RuntimeShape& input_shape,
                    const int8_t* input_data,
                    const tflite::RuntimeShape& output_shape,
                    OutputT* output_data)
{
    const int32_t input_beta_multiplier = params.input_multiplier;
    const int32_t input_beta_left_shift = params.input_left_shift;
    const int diff_min = params.diff_min;
    static const int kAccumulationIntegerBits = 12;

    const int trailing_dim = input_shape.DimensionsCount() - 1;
    const int outer_size = tflite::MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
    const int depth = tflite::MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
    const size_t depth_sz = static_cast<size_t>(depth);

    for (int i = 0; i < outer_size; ++i)
    {
        const int8_t* current_input_data = input_data + i * depth;
        OutputT* current_output_data = output_data + i * depth;

        int8_t max_in_row = std::numeric_limits<int8_t>::min();
        size_t current_c = 0;
        size_t vl_max_init = __riscv_vsetvl_e8m1(1);
        vint8m1_t v_max_acc_m1 = __riscv_vmv_v_x_i8m1(max_in_row, vl_max_init);
        while (current_c < depth_sz)
        {
            size_t vl = __riscv_vsetvl_e8m1(depth_sz - current_c);
            vint8m1_t v_input_m1 = __riscv_vle8_v_i8m1(current_input_data + current_c, vl);
            v_max_acc_m1 = __riscv_vredmax_vs_i8m1_i8m1(v_input_m1, v_max_acc_m1, vl);
            current_c += vl;
        }
        max_in_row = __riscv_vmv_x_s_i8m1_i8(v_max_acc_m1);
        const int32_t max_in_row_s32 = static_cast<int32_t>(max_in_row);

        size_t vl_sum_init = __riscv_vsetvl_e32m1(1);
        vint32m1_t v_sum_acc_m1 = __riscv_vmv_v_x_i32m1(0, vl_sum_init);
        current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl_m4 = __riscv_vsetvl_e32m4(depth_sz - current_c);
            size_t vl_m2 = __riscv_vsetvl_e16m2(vl_m4);
            size_t vl_m1 = __riscv_vsetvl_e8m1(vl_m4);

            vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(current_input_data + current_c, vl_m1);
            vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl_m2);
            vint32m4_t v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl_m4);
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl_m4);

            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl_m4);

            vint32m4_t v_diff_rescaled;
            {
              vint32m4_t v_a = __riscv_vsll_vx_i32m4(v_diff_s32, input_beta_left_shift, vl_m4);
              const int32_t b = input_beta_multiplier;
              v_diff_rescaled = SRDMH_vx_i32m4(v_a, b, vl_m4);
            }

            vint32m4_t v_exp_val_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled, vl_m4);

            const int rescale_shift = kAccumulationIntegerBits;
            vint32m4_t v_exp_term_q12_19 = SRMPOT_vx_i32m4(v_exp_val_q0_31, rescale_shift, vl_m4);

            vint32m4_t v_zero_q12_19 = __riscv_vmv_v_x_i32m4(0, vl_m4);
            vint32m4_t v_exp_term_masked_q12_19 = __riscv_vmerge_vvm_i32m4(v_zero_q12_19, v_exp_term_q12_19, v_diff_mask, vl_m4);

            v_sum_acc_m1 = __riscv_vredsum_vs_i32m4_i32m1(v_exp_term_masked_q12_19, v_sum_acc_m1, vl_m4);

            current_c += vl_m4;
        }
        int32_t sum_of_exps_raw = __riscv_vmv_x_s_i32m1_i32(v_sum_acc_m1);

        int num_bits_over_unit;
        gemmlowp::FixedPoint<int32_t, 0> shifted_scale = gemmlowp::FixedPoint<int32_t, 0>::FromRaw(tflite::GetReciprocal(sum_of_exps_raw, kAccumulationIntegerBits, &num_bits_over_unit));
        const int32_t s_shifted_scale_raw = shifted_scale.raw();

        const int exponent = num_bits_over_unit + 31 - (sizeof(OutputT) * 8);

        const OutputT output_min = std::numeric_limits<OutputT>::min();
        const OutputT output_max = std::numeric_limits<OutputT>::max();
        const int32_t output_min_s32 = static_cast<int32_t>(output_min);
        const int32_t output_max_s32 = static_cast<int32_t>(output_max);

        current_c = 0;
        while (current_c < depth_sz)
        {
            size_t vl_output;
             if constexpr (sizeof(OutputT) == 1)
             {
                vl_output = __riscv_vsetvl_e8m1(depth_sz - current_c);
             }
             else
             {
                vl_output = __riscv_vsetvl_e16m2(depth_sz - current_c);
             }
            size_t vl_m4 = __riscv_vsetvl_e32m4(vl_output);
            size_t vl_m2 = __riscv_vsetvl_e16m2(vl_output);
            size_t vl_m1 = __riscv_vsetvl_e8m1(vl_output);

            vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(current_input_data + current_c, vl_m1);
            vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl_m2);
            vint32m4_t v_input_s32 = __riscv_vwadd_vx_i32m4(v_input_s16, 0, vl_m4);
            vint32m4_t v_diff_s32 = __riscv_vsub_vx_i32m4(v_input_s32, max_in_row_s32, vl_m4);

            vbool8_t v_diff_mask = __riscv_vmsge_vx_i32m4_b8(v_diff_s32, diff_min, vl_m4);

            vint32m4_t v_diff_rescaled;
            {
              vint32m4_t v_a = __riscv_vsll_vx_i32m4(v_diff_s32, input_beta_left_shift, vl_m4);
              const int32_t b = input_beta_multiplier;
              v_diff_rescaled = SRDMH_vx_i32m4(v_a, b, vl_m4);
            }

            vint32m4_t v_exp_in_q0_31 = vectorized_exp_on_negative_values(v_diff_rescaled, vl_m4);

            vint32m4_t v_product_raw_q0_31 = RoundingMul_vx_i32m4(v_exp_in_q0_31, s_shifted_scale_raw, vl_m4);

            vint32m4_t v_zero_q0_31 = __riscv_vmv_v_x_i32m4(0, vl_m4);
            vint32m4_t v_product_masked = __riscv_vmerge_vvm_i32m4(v_zero_q0_31, v_product_raw_q0_31, v_diff_mask, vl_m4);

            vint32m4_t v_unsat_output;
            if (exponent <= 0)
            {
                v_unsat_output = v_product_masked;
            }
            else
            {
                const int32_t round_mask = (static_cast<int32_t>(1) << exponent) - 1;
                const int32_t threshold_base = round_mask >> 1;
                vint32m4_t v_x_shifted = __riscv_vsra_vx_i32m4(v_product_masked, exponent, vl_m4);
                vint32m4_t v_remainder = __riscv_vand_vx_i32m4(v_product_masked, round_mask, vl_m4);
                vbool8_t v_is_negative_mask = __riscv_vmslt_vx_i32m4_b8(v_product_masked, 0, vl_m4);

                vint32m4_t v_zero = __riscv_vmv_v_x_i32m4(0, vl_m4);
                vint32m4_t v_neg_adjust = __riscv_vmerge_vxm_i32m4(v_zero, 1, v_is_negative_mask, vl_m4);
                vint32m4_t v_threshold = __riscv_vadd_vx_i32m4(v_neg_adjust, threshold_base, vl_m4);

                vbool8_t v_P_mask = __riscv_vmsgt_vv_i32m4_b8(v_remainder, v_threshold, vl_m4);
                v_unsat_output = __riscv_vadd_vx_i32m4_m(v_P_mask, v_x_shifted, 1, vl_m4);
            }

            vint32m4_t v_shifted_output = __riscv_vadd_vx_i32m4(v_unsat_output, output_min_s32, vl_m4);
            vint32m4_t v_clamped_output = __riscv_vmax_vx_i32m4(v_shifted_output, output_min_s32, vl_m4);
            v_clamped_output = __riscv_vmin_vx_i32m4(v_clamped_output, output_max_s32, vl_m4);
            vint32m4_t v_final_s32 = v_clamped_output;


            if constexpr (sizeof(OutputT) == 1)
            {
                size_t vl_w16_out = __riscv_vsetvl_e16m2(vl_output);
                vint16m2_t v_temp_s16 = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl_w16_out);
                size_t vl_w8_out = __riscv_vsetvl_e8m1(vl_output);
                vint8m1_t v_final_output = __riscv_vncvt_x_x_w_i8m1(v_temp_s16, vl_w8_out);
                __riscv_vse8_v_i8m1(reinterpret_cast<int8_t*>(current_output_data + current_c), v_final_output, vl_output);
            }
            else
            {
                size_t vl_w16_out = __riscv_vsetvl_e16m2(vl_output);
                vint16m2_t v_final_output = __riscv_vncvt_x_x_w_i16m2(v_final_s32, vl_w16_out);
                __riscv_vse16_v_i16m2(reinterpret_cast<int16_t*>(current_output_data + current_c), v_final_output, vl_output);
            }
            
            current_c += vl_output;
        }
    }
}