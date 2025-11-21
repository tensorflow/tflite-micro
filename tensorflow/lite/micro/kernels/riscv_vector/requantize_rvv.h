#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_REQUANTIZE_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_REQUANTIZE_RVV_H_

inline vint32m2_t RequantizeVectorPerTensorS32(
    vint32m2_t v_acc, const int32_t multiplier, const int effective_right_shift,
    const int32_t output_offset, const int32_t activation_min,
    const int32_t activation_max, const size_t vl)
{
    // Calculate rounding constants for the 64-bit shift
    const int64_t rounding_val =
        (effective_right_shift > 0)
            ? (INT64_C(1) << (effective_right_shift - 1))
            : 0;
    const int32_t rounding_lo = static_cast<int32_t>(rounding_val);
    const int32_t rounding_hi = static_cast<int32_t>((rounding_val >> 32));

    // Multiply accumulator by scalar multiplier (results in 64b intermediate)
    // Uses m2 intrinsics
    vint32m2_t v_prod_lo = __riscv_vmul_vx_i32m2(v_acc, multiplier, vl);
    vint32m2_t v_prod_hi = __riscv_vmulh_vx_i32m2(v_acc, multiplier, vl);

    // Add 64b rounding value using 32b operations with carry
    vuint32m2_t v_prod_lo_u = __riscv_vreinterpret_v_i32m2_u32m2(v_prod_lo);
    vuint32m2_t v_sum_lo_u = __riscv_vadd_vx_u32m2(v_prod_lo_u, rounding_lo, vl);
    vbool16_t v_carry = __riscv_vmsltu_vx_u32m2_b16(v_sum_lo_u, rounding_lo, vl);
    vint32m2_t v_rounded_hi = __riscv_vadd_vx_i32m2(v_prod_hi, rounding_hi, vl);
    v_rounded_hi = __riscv_vadd_vx_i32m2_m(v_carry, v_rounded_hi, 1, vl);
    vint32m2_t v_rounded_lo = __riscv_vreinterpret_v_u32m2_i32m2(v_sum_lo_u);

    // Perform 64b arithmetic right shift using 32b vector shifts
    vint32m2_t v_res32;
    if (effective_right_shift == 0)
    {
        v_res32 = v_rounded_lo;
    }
    else if (effective_right_shift > 0 && effective_right_shift < 32)
    {
        vuint32m2_t v_lo_usrl = __riscv_vsrl_vx_u32m2(
            __riscv_vreinterpret_v_i32m2_u32m2(v_rounded_lo),
            effective_right_shift, vl);
        vint32m2_t v_hi_sll = __riscv_vsll_vx_i32m2(
            v_rounded_hi, 32 - effective_right_shift, vl);
        v_res32 = __riscv_vreinterpret_v_u32m2_i32m2(__riscv_vor_vv_u32m2(
            v_lo_usrl, __riscv_vreinterpret_v_i32m2_u32m2(v_hi_sll), vl));
    }
    else
    {
        const int shift_hi = std::min(31, effective_right_shift - 32);
        v_res32 = __riscv_vsra_vx_i32m2(v_rounded_hi, shift_hi, vl);
    }

    // Add output offset
    v_res32 = __riscv_vadd_vx_i32m2(v_res32, output_offset, vl);

    // Clamp to activation bounds
    v_res32 = __riscv_vmax_vx_i32m2(v_res32, activation_min, vl);
    v_res32 = __riscv_vmin_vx_i32m2(v_res32, activation_max, vl);

    return v_res32;
}

inline vint32m2_t RequantizeVectorPerChannelS32(
    vint32m2_t v_acc, vint32m2_t v_multiplier, vint32m2_t v_shift,
    const int32_t output_offset, const int32_t activation_min,
    const int32_t activation_max, const size_t vl)
{
    // Perform 32x32 -> 64-bit multiplication
    vint32m2_t v_prod_hi = __riscv_vmulh_vv_i32m2(v_acc, v_multiplier, vl);
    vint32m2_t v_prod_lo = __riscv_vmul_vv_i32m2(v_acc, v_multiplier, vl);

    // Calculate effective right shift
    vint32m2_t v_effective_shift = __riscv_vrsub_vx_i32m2(v_shift, 31, vl);

    // Create masks
    vbool16_t v_mask_right_shift =
        __riscv_vmsgt_vx_i32m2_b16(v_effective_shift, 0, vl);
    vbool16_t v_mask_left_shift = __riscv_vmnot_m_b16(v_mask_right_shift, vl);

    // Path 1: Right Shift
    // Initialize to 0 to avoid "maybe-uninitialized" warnings
    vint32m2_t v_res_right = __riscv_vmv_v_x_i32m2(0, vl);
    
    // Optimization: check if any lane needs right shift
    if (__riscv_vfirst_m_b16(v_mask_right_shift, vl) >= 0) 
    {
        vint32m2_t v_shift_minus_1 = __riscv_vsub_vx_i32m2_m(
            v_mask_right_shift, v_effective_shift, 1, vl);
        vuint32m2_t v_shift_minus_1_u =
            __riscv_vreinterpret_v_i32m2_u32m2(v_shift_minus_1);
        vbool16_t v_mask_round_lt_32 = __riscv_vmsltu_vx_u32m2_b16_m(
            v_mask_right_shift, v_shift_minus_1_u, 32, vl);
        vbool16_t v_mask_round_ge_32 = __riscv_vmandn_mm_b16(
            v_mask_right_shift, v_mask_round_lt_32, vl);
        vuint32m2_t v_one_u = __riscv_vmv_v_x_u32m2(1, vl);
        vuint32m2_t v_zero_u = __riscv_vmv_v_x_u32m2(0, vl);
        vuint32m2_t v_rounding_lo_u = __riscv_vmerge_vvm_u32m2(
            v_zero_u,
            __riscv_vsll_vv_u32m2_m(v_mask_round_lt_32, v_one_u,
                                   v_shift_minus_1_u, vl),
            v_mask_round_lt_32, vl);
        vuint32m2_t v_rounding_hi_u = __riscv_vmerge_vvm_u32m2(
            v_zero_u,
            __riscv_vsll_vv_u32m2_m(
                v_mask_round_ge_32, v_one_u,
                __riscv_vsub_vx_u32m2_m(v_mask_round_ge_32, v_shift_minus_1_u,
                                        32, vl),
                vl),
            v_mask_round_ge_32, vl);

        vuint32m2_t v_prod_lo_u = __riscv_vreinterpret_v_i32m2_u32m2(v_prod_lo);
        vuint32m2_t v_sum_lo_u = __riscv_vadd_vv_u32m2_m(
            v_mask_right_shift, v_prod_lo_u, v_rounding_lo_u, vl);
        vbool16_t v_carry = __riscv_vmsltu_vv_u32m2_b16_m(
            v_mask_right_shift, v_sum_lo_u, v_prod_lo_u, vl);
        vint32m2_t v_rounded_hi = __riscv_vadd_vv_i32m2_m(
            v_mask_right_shift, v_prod_hi,
            __riscv_vreinterpret_v_u32m2_i32m2(v_rounding_hi_u), vl);
        v_rounded_hi = __riscv_vadd_vx_i32m2_m(v_carry, v_rounded_hi, 1, vl);

        vbool16_t v_mask_shift_lt_32 = __riscv_vmslt_vx_i32m2_b16_m(
            v_mask_right_shift, v_effective_shift, 32, vl);
        vbool16_t v_mask_shift_ge_32 = __riscv_vmandn_mm_b16(
            v_mask_right_shift, v_mask_shift_lt_32, vl);
        vuint32m2_t v_shift_u =
            __riscv_vreinterpret_v_i32m2_u32m2(v_effective_shift);
        vuint32m2_t v_lo_part = __riscv_vsrl_vv_u32m2_m(
            v_mask_shift_lt_32, v_sum_lo_u, v_shift_u, vl);
        vuint32m2_t v_hi_part = __riscv_vsll_vv_u32m2_m(
            v_mask_shift_lt_32,
            __riscv_vreinterpret_v_i32m2_u32m2(v_rounded_hi),
            __riscv_vrsub_vx_u32m2_m(v_mask_shift_lt_32, v_shift_u, 32, vl),
            vl);
        vint32m2_t v_res_lt_32 = __riscv_vreinterpret_v_u32m2_i32m2(
            __riscv_vor_vv_u32m2_m(v_mask_shift_lt_32, v_lo_part, v_hi_part, vl));
        vint32m2_t v_res_ge_32 = __riscv_vsra_vv_i32m2_m(
            v_mask_shift_ge_32, v_rounded_hi,
            __riscv_vreinterpret_v_i32m2_u32m2(__riscv_vsub_vx_i32m2_m(
                v_mask_shift_ge_32, v_effective_shift, 32, vl)),
            vl);
        v_res_right = __riscv_vmerge_vvm_i32m2(v_res_ge_32, v_res_lt_32,
                                              v_mask_shift_lt_32, vl);
    }

    // Path 2: Left Shift
    // Initialize to 0 to avoid "maybe-uninitialized" warnings
    vint32m2_t v_res_left = __riscv_vmv_v_x_i32m2(0, vl);
    
    if (__riscv_vfirst_m_b16(v_mask_left_shift, vl) >= 0)
    {
        vint32m2_t v_left_shift_amount =
            __riscv_vneg_v_i32m2_m(v_mask_left_shift, v_effective_shift, vl);

        v_res_left = __riscv_vsll_vv_i32m2_m(
            v_mask_left_shift, v_prod_lo,
            __riscv_vreinterpret_v_i32m2_u32m2(v_left_shift_amount), vl);
    }

    // Merge results
    // Lanes with mask_right=1 take v_res_right, mask_right=0 (left) take v_res_left
    vint32m2_t v_res32 =
        __riscv_vmerge_vvm_i32m2(v_res_left, v_res_right, v_mask_right_shift, vl);

    // Add output offset
    v_res32 = __riscv_vadd_vx_i32m2(v_res32, output_offset, vl);

    // Clamp to activation bounds
    v_res32 = __riscv_vmax_vx_i32m2(v_res32, activation_min, vl);
    v_res32 = __riscv_vmin_vx_i32m2(v_res32, activation_max, vl);

    return v_res32;
}

#endif