#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

using namespace tflite;

void FullyConnectedPerChannelRVV(const tflite::FullyConnectedParams& params,
                                 const int32_t* output_multiplier,
                                 const int* output_shift,
                                 const tflite::RuntimeShape& input_shape,
                                 const int8_t* input_data,
                                 const tflite::RuntimeShape& filter_shape,
                                 const int8_t* filter_data,
                                 const tflite::RuntimeShape& bias_shape,
                                 const int32_t* bias_data,
                                 const tflite::RuntimeShape& output_shape,
                                 int8_t* output_data) {
  // Extract quantization parameters
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Extract shape dimensions
  const int batches = FlatSizeSkipDim(output_shape, output_shape.DimensionsCount() - 1);
  const int output_depth = output_shape.Dims(output_shape.DimensionsCount() - 1);
  const int accum_depth = filter_shape.Dims(filter_shape.DimensionsCount() - 1);

  // Prepare scalar constants
  const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);

  // Loop over batches
  for (int b = 0; b < batches; ++b) {
    const int8_t* input_batch_ptr = input_data + b * accum_depth;
    int8_t* output_batch_ptr = output_data + b * output_depth;

    // Vectorized loop over output channels
    size_t current_out_c = 0;
    while (current_out_c < static_cast<size_t>(output_depth)) {
      // Set vector length for this iteration
      size_t vl = __riscv_vsetvl_e32m4(output_depth - current_out_c);

      // Initialize accumulator vector with biases
      vint32m4_t v_acc_s32;
      if (bias_data) {
        v_acc_s32 = __riscv_vle32_v_i32m4(bias_data + current_out_c, vl);
      } else {
        v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vl);
      }

      // Main MAC loop to compute dot products
      for (int d = 0; d < accum_depth; ++d) {
        // Load scalar input value and add offset
        int16_t s_input_val_s16 = static_cast<int16_t>(input_batch_ptr[d]) + s_input_offset_s16;
        
        // Calculate filter pointer and stride for the current column
        const int8_t* filter_col_ptr = filter_data + d + current_out_c * accum_depth;
        ptrdiff_t filter_stride = accum_depth * sizeof(int8_t);

        // Load filter vector, widen, and perform widening multiply-accumulate
        vint8m1_t v_filter_s8 = __riscv_vlse8_v_i8m1(filter_col_ptr, filter_stride, vl);
        vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);
        v_acc_s32 = __riscv_vwmacc_vx_i32m4(v_acc_s32, s_input_val_s16, v_filter_s16, vl);
      }

      // Start of fully vectorized per-channel requantization
      vint32m4_t v_res32;
      
      // Load per-channel requantization parameters into vectors
      vint32m4_t v_multiplier = __riscv_vle32_v_i32m4(output_multiplier + current_out_c, vl);
      vint32m4_t v_shift = __riscv_vle32_v_i32m4(reinterpret_cast<const int32_t*>(output_shift) + current_out_c, vl);

      // Create a mask for lanes that require a right shift (where shift > 0)
      vbool8_t v_mask_right_shift = __riscv_vmsgt_vx_i32m4_b8(v_shift, 0, vl);

      // Path 1: Right Shift (for lanes where shift > 0)
      vint32m4_t v_res_right;
      {
        // Calculate the 64-bit product of accumulator and multiplier
        vint32m4_t v_prod_hi = __riscv_vmulh_vv_i32m4_m(v_mask_right_shift, v_acc_s32, v_multiplier, vl);
        vint32m4_t v_prod_lo = __riscv_vmul_vv_i32m4_m(v_mask_right_shift, v_acc_s32, v_multiplier, vl);

        // Calculate the 64-bit rounding value: (1 << (shift - 1))
        vint32m4_t v_shift_minus_1 = __riscv_vsub_vx_i32m4_m(v_mask_right_shift, v_shift, 1, vl);
        vuint32m4_t v_one_u = __riscv_vmv_v_x_u32m4(1, vl);
        vuint32m4_t v_rounding_u = __riscv_vsll_vv_u32m4_m(v_mask_right_shift, v_one_u, __riscv_vreinterpret_v_i32m4_u32m4(v_shift_minus_1), vl);

        // Add the 64-bit rounding value to the 64-bit product
        vuint32m4_t v_prod_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
        vuint32m4_t v_sum_lo_u = __riscv_vadd_vv_u32m4_m(v_mask_right_shift, v_prod_lo_u, v_rounding_u, vl);
        vbool8_t v_carry = __riscv_vmsltu_vv_u32m4_b8_m(v_mask_right_shift, v_sum_lo_u, v_prod_lo_u, vl);
        vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_prod_hi, 1, vl);

        // Create a mask to select between the two 64-bit shift emulation paths
        vbool8_t v_mask_shift_lt_32 = __riscv_vmslt_vx_i32m4_b8_m(v_mask_right_shift, v_shift, 32, vl);

        // Sub-path A: Emulate 64-bit shift for 0 < shift < 32
        vint32m4_t v_res_lt_32;
        {
          vuint32m4_t v_shift_u = __riscv_vreinterpret_v_i32m4_u32m4(v_shift);
          vuint32m4_t v_shift_rev_u = __riscv_vrsub_vx_u32m4_m(v_mask_shift_lt_32, v_shift_u, 32, vl);
          vuint32m4_t v_lo_part = __riscv_vsrl_vv_u32m4_m(v_mask_shift_lt_32, v_sum_lo_u, v_shift_u, vl);
          vuint32m4_t v_hi_part = __riscv_vsll_vv_u32m4_m(v_mask_shift_lt_32, __riscv_vreinterpret_v_i32m4_u32m4(v_rounded_hi), v_shift_rev_u, vl);
          v_res_lt_32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4_m(v_mask_shift_lt_32, v_lo_part, v_hi_part, vl));
        }

        // Sub-path B: Emulate 64-bit shift for shift >= 32
        vint32m4_t v_res_ge_32;
        {
          vbool8_t v_mask_shift_ge_32 = __riscv_vmandn_mm_b8(v_mask_right_shift, v_mask_shift_lt_32, vl);
          vint32m4_t v_shift_hi = __riscv_vsub_vx_i32m4_m(v_mask_shift_ge_32, v_shift, 32, vl);
          v_shift_hi = __riscv_vmin_vx_i32m4_m(v_mask_shift_ge_32, v_shift_hi, 31, vl); // Clamp to 31
          v_res_ge_32 = __riscv_vsra_vv_i32m4_m(v_mask_shift_ge_32, v_rounded_hi, __riscv_vreinterpret_v_i32m4_u32m4(v_shift_hi), vl);
        }

        // Merge the results from the two 64-bit shift sub-paths
        v_res_right = __riscv_vmerge_vvm_i32m4(v_res_ge_32, v_res_lt_32, v_mask_shift_lt_32, vl);
      }

      // Path 2: Left Shift (for lanes where shift <= 0)
      vint32m4_t v_res_left;
      {
        // Negate the shift amount and perform a left shift on the accumulator
        vint32m4_t v_neg_shift = __riscv_vneg_v_i32m4(v_shift, vl);
        v_res_left = __riscv_vsll_vv_i32m4(v_acc_s32, __riscv_vreinterpret_v_i32m4_u32m4(v_neg_shift), vl);
      }

      // Merge the results from the right and left shift paths
      v_res32 = __riscv_vmerge_vvm_i32m4(v_res_left, v_res_right, v_mask_right_shift, vl);

      // Add the final output offset
      v_res32 = __riscv_vadd_vx_i32m4(v_res32, output_offset, vl);

      // Clamp the results to the activation range
      v_res32 = __riscv_vmax_vx_i32m4(v_res32, output_activation_min, vl);
      v_res32 = __riscv_vmin_vx_i32m4(v_res32, output_activation_max, vl);

      // Narrow the 32-bit results to 16-bit, then 8-bit with saturation
      vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
      vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);
      
      // Store the final 8-bit output vector
      __riscv_vse8_v_i8m1(output_batch_ptr + current_out_c, v_out_s8, vl);

      // Advance to the next block of output channels
      current_out_c += vl;
    }
  }
}

void FullyConnectedRVV(const FullyConnectedParams& params,
                  const RuntimeShape& input_shape,
                  const int8_t* input_data,
                  const RuntimeShape& filter_shape,
                  const int8_t* filter_data,
                  const RuntimeShape& bias_shape,
                  const int32_t* bias_data,
                  const RuntimeShape& output_shape,
                  int8_t* output_data)
{
    // Extract quantization parameters
    const int32_t input_offset = params.input_offset;
    const int32_t filter_offset = params.weights_offset;
    const int32_t output_offset = params.output_offset;
    const int32_t output_multiplier = params.output_multiplier;
    const int output_shift = params.output_shift;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // Extract shape dimensions
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int output_dim_count = output_shape.DimensionsCount();
    const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
    const int output_depth = output_shape.Dims(output_dim_count - 1);
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

    // Prepare scalar constants for vector operations
    const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);
    const int16_t s_filter_offset_s16 = static_cast<int16_t>(filter_offset);

    // Loop over batches
    for (int b = 0; b < batches; ++b)
    {
        const int8_t* input_batch_ptr = input_data + b * accum_depth;
        int8_t* output_batch_ptr = output_data + b * output_depth;

        // Vectorized loop over output channels
        size_t current_out_c = 0;
        while (current_out_c < static_cast<size_t>(output_depth)) {
            // Set vector length for processing multiple output channels
            size_t vl = __riscv_vsetvl_e32m4(output_depth - current_out_c);

            // Initialize accumulator vector with biases
            vint32m4_t v_acc_s32;
            if (bias_data) {
                v_acc_s32 = __riscv_vle32_v_i32m4(bias_data + current_out_c, vl);
            } else {
                v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vl);
            }

            // Loop over accumulation depth to compute 'vl' dot products in parallel
            for (int d = 0; d < accum_depth; ++d) {
                // Load one scalar from the input vector and add offset
                int16_t s_input_val_s16 = static_cast<int16_t>(input_batch_ptr[d]) + s_input_offset_s16;

                // Load a vector of 'vl' filter values (a column slice)
                const int8_t* filter_col_ptr = filter_data + current_out_c * accum_depth + d;
                ptrdiff_t filter_stride = accum_depth * sizeof(int8_t);
                vint8m1_t v_filter_s8 = __riscv_vlse8_v_i8m1(filter_col_ptr, filter_stride, vl);
                
                // Widen filter values and add filter offset
                vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);
                vint16m2_t v_filter_plus_offset_s16 = __riscv_vadd_vx_i16m2(v_filter_s16, s_filter_offset_s16, vl);
                
                // Perform widening vector-scalar multiply-accumulate
                v_acc_s32 = __riscv_vwmacc_vx_i32m4(v_acc_s32, s_input_val_s16, v_filter_plus_offset_s16, vl);
            }

            // Start of inline vectorized requantization
            vint32m4_t v_res32;
            const int effective_right_shift = 31 - output_shift;

            // Calculate rounding constants
            int64_t rounding_val = (effective_right_shift > 0) ? (INT64_C(1) << (effective_right_shift - 1)) : 0;
            int32_t rounding_lo = static_cast<int32_t>(rounding_val);
            int32_t rounding_hi = static_cast<int32_t>((rounding_val >> 32));

            // Multiply accumulator by scalar multiplier (results in 64b intermediate)
            vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_acc_s32, output_multiplier, vl);
            vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_acc_s32, output_multiplier, vl);

            // Add 64b rounding value
            vuint32m4_t v_acc_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
            vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_acc_lo_u, rounding_lo, vl);
            vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding_lo, vl);
            vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, rounding_hi, vl);
            v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
            vint32m4_t v_rounded_lo = __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_u);

            // Perform 64b arithmetic right shift
            if (effective_right_shift == 0) {
                v_res32 = v_rounded_lo;
            } else if (effective_right_shift > 0 && effective_right_shift < 32) {
                vuint32m4_t v_lo_usrl = __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(v_rounded_lo), effective_right_shift, vl);
                vint32m4_t v_hi_sll = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_right_shift, vl);
                v_res32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4(v_lo_usrl, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_sll), vl));
            } else {
                int shift_hi = std::min(31, effective_right_shift - 32);
                v_res32 = __riscv_vsra_vx_i32m4(v_rounded_hi, shift_hi, vl);
            }

            // Add output offset
            v_res32 = __riscv_vadd_vx_i32m4(v_res32, output_offset, vl);

            // Clamp to activation bounds
            v_res32 = __riscv_vmax_vx_i32m4(v_res32, output_activation_min, vl);
            v_res32 = __riscv_vmin_vx_i32m4(v_res32, output_activation_max, vl);

            // Narrow result to int8 and store
            vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
            vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_i8m1(output_batch_ptr + current_out_c, v_out_s8, vl);

            // Advance to the next block of output channels
            current_out_c += vl;
        }
    }
}