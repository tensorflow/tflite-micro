#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <limits> 

#include <riscv_vector.h>

__attribute__((hot))
void convolution_hwc_ohwi_rvv(
    const uint8_t* input_data,
    const uint16_t input_height,
    const uint16_t input_width,
    const uint16_t input_channels,
    const int32_t input_offset,
    const int8_t* filter_data,
    const uint16_t filter_height,
    const uint16_t filter_width,
    const int32_t* bias_data,
    uint8_t* output_data,
    const uint16_t output_height,
    const uint16_t output_width,
    const uint16_t output_channels,
    const int32_t output_offset,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const uint16_t stride_height,
    const uint16_t stride_width,
    const uint16_t pad_height,
    const uint16_t pad_width)
{
    assert(input_data != nullptr);
    assert(filter_data != nullptr);
    assert(bias_data != nullptr);
    assert(output_data != nullptr);
    assert(output_multiplier != nullptr);
    assert(output_shift != nullptr);
    assert(input_height > 0); 
    assert(input_width > 0); 
    assert(input_channels > 0);
    assert(filter_height > 0); 
    assert(filter_width > 0);
    assert(output_height > 0); 
    assert(output_width > 0); 
    assert(output_channels > 0);
    assert(stride_height > 0); 
    assert(stride_width > 0);
    assert(input_offset >= 0 && input_offset <= 255);

    // Pre-calculate strides and kernel plane size for efficient access
    const size_t input_row_stride = (size_t)input_width * input_channels;
    const size_t output_row_stride = (size_t)output_width * output_channels;
    const size_t filter_kernel_plane_size = (size_t)filter_height * filter_width * input_channels;

    // Define activation clamping limits based on output type
    const int32_t output_activation_min_i32 = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
    const int32_t output_activation_max_i32 = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());

    // Set the default rounding mode for fixed-point vector instructions
    const unsigned int default_vxrm = __RISCV_VXRM_RNU;

    // Iterate through each output channel
    for (int out_c = 0; out_c < output_channels; ++out_c)
    {
        // Calculate the sum of filter weights for the current channel for offset correction
        int32_t filter_sum = 0;
        const int8_t* filter_start_for_channel = filter_data + (size_t)out_c * filter_kernel_plane_size;
        for (size_t i = 0; i < filter_kernel_plane_size; ++i)
        {
            filter_sum += filter_start_for_channel[i];
        }

        // Pre-calculate per-channel constants for bias, quantization, and correction
        const int32_t current_offset_correction = input_offset * filter_sum;
        const int32_t current_bias = bias_data[out_c];
        const int32_t current_output_multiplier = output_multiplier[out_c];
        const int32_t current_output_shift = output_shift[out_c];

        // Determine requantization shifts and rounding offset
        const int32_t left_shift = std::max((int32_t)0, current_output_shift);
        const int32_t right_shift = std::max((int32_t)0, -current_output_shift);
        const int32_t rounding_offset = (right_shift > 0) ? (1 << (right_shift - 1)) : 0;

        // Calculate saturation limits for intermediate requantization steps
        const int32_t add_rounding_limit = INT32_MAX - rounding_offset;
        const int32_t add_offset_limit_pos = INT32_MAX - output_offset;
        const int32_t add_offset_limit_neg = INT32_MIN - output_offset;
        const int32_t left_shift_limit_pos = (left_shift < 31) ? (INT32_MAX >> left_shift) : 0;
        const int32_t left_shift_limit_neg = (left_shift < 31) ? (INT32_MIN >> left_shift) : -1;

        // Iterate through each output row
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            // Calculate the starting input row corresponding to the current output row
            const int in_y_origin = (out_y * stride_height) - pad_height;

            // Process output row strip-by-strip
            size_t current_out_x = 0;
            while (current_out_x < output_width)
            {
                // Set the vector length for the current strip
                const size_t vl = __riscv_vsetvl_e32m8(output_width - current_out_x);

                // Initialize the accumulator vector with the bias for the current channel
                vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(current_bias, vl);

                // Iterate through the filter kernel height
                for (int k_y = 0; k_y < filter_height; ++k_y)
                {
                    // Calculate the current input row index and skip if out of bounds (padding)
                    const int in_y = in_y_origin + k_y;
                    if (in_y < 0 || in_y >= input_height) continue;

                    // Iterate through the filter kernel width
                    for (int k_x = 0; k_x < filter_width; ++k_x)
                    {
                        // Calculate the vector of input x coordinates for the current strip
                        const int32_t in_x_origin_for_k = (int32_t)k_x - pad_width;
                        vuint32m8_t v_lane_indices = __riscv_vid_v_u32m8(vl);
                        vuint32m8_t v_out_x_indices = __riscv_vadd_vx_u32m8(v_lane_indices, current_out_x, vl);
                        vuint32m8_t v_in_x_base = __riscv_vmul_vx_u32m8(v_out_x_indices, stride_width, vl);
                        vuint32m8_t v_in_x_u32_temp = __riscv_vadd_vx_u32m8(v_in_x_base, in_x_origin_for_k, vl);
                        vint32m8_t v_in_x_i32 = __riscv_vreinterpret_v_u32m8_i32m8(v_in_x_u32_temp);

                        // Generate a mask for valid input x coordinates (handling horizontal padding)
                        vbool4_t v_mask_x_ge_0 = __riscv_vmsge_vx_i32m8_b4(v_in_x_i32, 0, vl);
                        vbool4_t v_mask_x_lt_w = __riscv_vmslt_vx_i32m8_b4(v_in_x_i32, input_width, vl);
                        vbool4_t v_mask_valid_x = __riscv_vmand_mm_b4(v_mask_x_ge_0, v_mask_x_lt_w, vl);

                        // Calculate the input x offset scaled by the number of channels
                        vuint32m8_t v_in_x_u32 = __riscv_vreinterpret_v_i32m8_u32m8(v_in_x_i32);
                        vuint32m8_t v_in_x_ch_offset = __riscv_vmul_vx_u32m8(v_in_x_u32, input_channels, vl);

                        // Get the base pointer for the filter weights for this kernel position
                        const int8_t* filter_ptr_base = filter_data +
                            (size_t)out_c * filter_kernel_plane_size +
                            (size_t)k_y * filter_width * input_channels +
                            (size_t)k_x * input_channels;

                        // Iterate through the input channels, performing MAC operations
                        for (int in_c = 0; in_c < input_channels; ++in_c)
                        {
                            // Skip MAC if filter weight is zero
                            const int8_t filter_val = filter_ptr_base[in_c];
                            if (filter_val == 0) 
                                continue;

                            // Calculate the vector of byte offsets into the input data
                            uint32_t base_offset_for_row_ch = (uint32_t)in_y * input_row_stride + in_c;
                            vuint32m8_t v_byte_offset_u32 = __riscv_vadd_vx_u32m8(v_in_x_ch_offset, base_offset_for_row_ch, vl);

                            // Load input data elements using indexed load
                            vuint8m2_t v_loaded_raw = __riscv_vloxei32_v_u8m2(
                                                            input_data,
                                                            v_byte_offset_u32,
                                                            vl);

                            // Create a vector of input zero-points
                            uint8_t input_zero_point_u8 = (uint8_t)input_offset;
                            vuint8m2_t v_zero_points = __riscv_vmv_v_x_u8m2(input_zero_point_u8, vl);

                            // Merge loaded data with zero-points based on the padding mask
                            vuint8m2_t v_input_u8 = __riscv_vmerge_vvm_u8m2(
                                                            v_loaded_raw,
                                                            v_zero_points,
                                                            v_mask_valid_x,
                                                            vl);

                            // Sign-extend input from 8-bit to 16-bit for widening MAC
                            vint8m2_t v_input_i8 = __riscv_vreinterpret_v_u8m2_i8m2(v_input_u8);
                            vint16m4_t v_input_i16 = __riscv_vsext_vf2_i16m4(v_input_i8, vl);

                            // Perform widening multiply-accumulate operation
                            v_acc = __riscv_vwmacc_vx_i32m8(v_acc,
                                                            filter_val,
                                                            v_input_i16,
                                                            vl);
                        }
                    }
                }

                // Apply the input offset correction term to the accumulator
                v_acc = __riscv_vsub_vx_i32m8(v_acc, current_offset_correction, vl);

                // Multiply by output multiplier (high part)
                vint32m8_t v_requant_stage1 = __riscv_vmulh_vx_i32m8(v_acc, current_output_multiplier, vl);

                // Declare a temporary mask variable for requantization saturation checks
                vbool4_t v_temp_mask_b4;

                // Apply rounding offset and right shift with saturation
                if (right_shift > 0) 
                {
                    v_temp_mask_b4 = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, add_rounding_limit, vl);
                    vint32m8_t v_added_round = __riscv_vadd_vx_i32m8(v_requant_stage1, rounding_offset, vl);
                    v_requant_stage1 = __riscv_vmerge_vxm_i32m8(v_added_round, INT32_MAX, v_temp_mask_b4, vl);
                    v_requant_stage1 = __riscv_vsra_vx_i32m8(v_requant_stage1, right_shift, vl);
                }

                // Apply left shift with saturation if needed
                if (left_shift > 0) 
                {
                    v_temp_mask_b4 = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, left_shift_limit_pos, vl);
                    vbool4_t v_temp_mask_b4_neg = __riscv_vmslt_vx_i32m8_b4(v_requant_stage1, left_shift_limit_neg, vl);
                    vint32m8_t v_shifted = __riscv_vsll_vx_i32m8(v_requant_stage1, left_shift, vl);
                    v_shifted = __riscv_vmerge_vxm_i32m8(v_shifted, INT32_MAX, v_temp_mask_b4, vl);
                    v_shifted = __riscv_vmerge_vxm_i32m8(v_shifted, INT32_MIN, v_temp_mask_b4_neg, vl);
                    v_requant_stage1 = v_shifted;
                }

                // Add output offset with saturation
                v_temp_mask_b4 = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, add_offset_limit_pos, vl);
                vbool4_t v_temp_mask_b4_neg = __riscv_vmslt_vx_i32m8_b4(v_requant_stage1, add_offset_limit_neg, vl);
                vint32m8_t v_requant_stage2 = __riscv_vadd_vx_i32m8(v_requant_stage1, output_offset, vl);
                v_requant_stage2 = __riscv_vmerge_vxm_i32m8(v_requant_stage2, INT32_MAX, v_temp_mask_b4, vl);
                v_requant_stage2 = __riscv_vmerge_vxm_i32m8(v_requant_stage2, INT32_MIN, v_temp_mask_b4_neg, vl);

                // Clamp the result to the final activation range [0, 255]
                vint32m8_t v_clamped_i32 = __riscv_vmax_vx_i32m8(v_requant_stage2, output_activation_min_i32, vl);
                v_clamped_i32 = __riscv_vmin_vx_i32m8(v_clamped_i32, output_activation_max_i32, vl);

                // Narrow the 32-bit results down to 8-bit unsigned integers
                vint16m4_t v_narrowed_i16 = __riscv_vnsra_wx_i16m4(v_clamped_i32, 0, vl);
                vuint16m4_t v_narrowed_u16 = __riscv_vreinterpret_v_i16m4_u16m4(v_narrowed_i16);
                vuint8m2_t v_output_u8 = __riscv_vnclipu_wx_u8m2(v_narrowed_u16, 0, default_vxrm, vl);

                // Calculate the base pointer for storing the output strip
                uint8_t* output_base_ptr = output_data +
                                            (size_t)out_y * output_row_stride +
                                            current_out_x * output_channels +
                                            out_c;

                // Define the byte stride for storing into the HWC output layout
                ptrdiff_t byte_stride = (ptrdiff_t)output_channels;

                // Store the computed 8-bit output values using strided store
                __riscv_vsse8_v_u8m2(output_base_ptr, byte_stride, v_output_u8, vl);

                // Move to the next horizontal strip
                current_out_x += vl;
            }
        }
    }
}