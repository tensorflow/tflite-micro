#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "requantize_rvv.h"

using namespace tflite;

void ConvPerChannelRVV(const ConvParams& params,
                  const int32_t* output_multiplier,
                  const int32_t* output_shift,
                  const RuntimeShape& input_shape,
                  const int8_t* input_data,
                  const RuntimeShape& filter_shape,
                  const int8_t* filter_data,
                  const RuntimeShape& bias_shape,
                  const int32_t* bias_data,
                  const RuntimeShape& output_shape,
                  int8_t* output_data)
{
    // Extract convolution parameters
    const int32_t input_offset = params.input_offset;
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // Extract shape dimensions
    const int input_batches = input_shape.Dims(0);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth = output_shape.Dims(3);

    // Calculate grouping parameters
    const int groups = input_depth / filter_input_depth;
    const int filters_per_group = output_depth / groups;

    // Calculate tensor strides
    const int input_ch_stride = 1;
    const int input_w_stride = input_depth;
    const int input_h_stride = input_width * input_w_stride;
    const int input_b_stride = input_height * input_h_stride;
    const int filter_ch_stride = 1;
    const int filter_w_stride = filter_input_depth;
    const int filter_h_stride = filter_width * filter_w_stride;
    const int filter_o_stride = filter_height * filter_h_stride;
    const int output_ch_stride = 1;
    const int output_w_stride = output_depth;
    const int output_h_stride = output_width * output_w_stride;
    const int output_b_stride = output_height * output_h_stride;

    // Prepare scalar constants
    const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    // Loop over batches
    for (int batch = 0; batch < input_batches; ++batch)
    {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;

        // Loop over output height
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int8_t* output_row_base = output_batch_base + out_y * output_h_stride;

            // Loop over output channels
            for (int out_channel = 0; out_channel < output_depth; ++out_channel)
            {
                // Calculate group and filter parameters for this output channel
                const int group = out_channel / filters_per_group;
                const int group_start_input_channel = group * filter_input_depth;
                const int8_t* filter_oc_base = filter_data + out_channel * filter_o_stride;

                // Get per-channel requantization parameters
                const int32_t scalar_multiplier = output_multiplier[out_channel];
                const int32_t scalar_shift = output_shift[out_channel];
                const int effective_right_shift = 31 - scalar_shift;

                // Get bias value for this output channel
                const int32_t bias_val = bias_data ? bias_data[out_channel] : 0;

                // Calculate output pointer and stride for this channel row
                int8_t* output_channel_base = output_row_base + out_channel * output_ch_stride;
                const ptrdiff_t output_x_stride_bytes = output_w_stride * sizeof(int8_t);

                // Process output width in vector chunks
                size_t current_out_x = 0;
                while (current_out_x < static_cast<size_t>(output_width))
                {
                    // Set vector length for this iteration
                    size_t vl = __riscv_vsetvl_e32m4(output_width - current_out_x);

                    // Initialize accumulator vector with bias
                    vint32m4_t v_acc_s32 = bias_data ? __riscv_vmv_v_x_i32m4(bias_val, vl)
                                                     : __riscv_vmv_v_x_i32m4(0, vl);

                    // Calculate base input x coordinates for the vector lanes
                    vuint32m4_t v_idx = __riscv_vid_v_u32m4(vl);
                    vint32m4_t v_out_x = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vx_u32m4(v_idx, static_cast<uint32_t>(current_out_x), vl));
                    vint32m4_t v_in_x_origin_base = __riscv_vsub_vx_i32m4(__riscv_vmul_vx_i32m4(v_out_x, stride_width, vl), pad_width, vl);

                    // Loop over filter height
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y)
                    {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        if (in_y < 0 || in_y >= input_height) continue; // Simplified boundary check

                        const int8_t* filter_y_base = filter_oc_base + (filter_y * filter_h_stride);

                        // Loop over filter width
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x)
                        {
                            const int in_x_offset = dilation_width_factor * filter_x;
                            const int8_t* filter_patch_base = filter_y_base + (filter_x * filter_w_stride);
                            vint32m4_t v_in_x = __riscv_vadd_vx_i32m4(v_in_x_origin_base, in_x_offset, vl);

                            // Create mask for valid input coordinates
                            vbool8_t v_mask_ge_zero = __riscv_vmsge_vx_i32m4_b8(v_in_x, 0, vl);
                            vbool8_t v_mask_lt_width = __riscv_vmslt_vx_i32m4_b8(v_in_x, input_width, vl);
                            vbool8_t v_active_lane_mask_b8 = __riscv_vmand_mm_b8(v_mask_ge_zero, v_mask_lt_width, vl);

                            // Calculate base input pointer and stride for vector load
                            int32_t base_in_x_for_vector0 = static_cast<int32_t>(current_out_x) * stride_width - pad_width + in_x_offset;
                            const int8_t* input_base_for_y_x_patch = input_batch_base + (in_y * input_h_stride) + (base_in_x_for_vector0 * input_w_stride) +
                                                                     (group_start_input_channel * input_ch_stride);
                            ptrdiff_t input_x_stride_bytes = static_cast<ptrdiff_t>(stride_width) * input_w_stride * sizeof(int8_t);

                            // Loop over input channels for this filter tap
                            for (int ic = 0; ic < filter_input_depth; ++ic)
                            {
                                int8_t s_filter_val_s8 = filter_patch_base[ic * filter_ch_stride];
                                int16_t s_filter_val_s16 = static_cast<int16_t>(s_filter_val_s8);
                                const int8_t* input_ic_ptr = input_base_for_y_x_patch + (ic * input_ch_stride);
                                vint8m1_t v_input_s8 = __riscv_vlse8_v_i8m1_m(v_active_lane_mask_b8, input_ic_ptr, input_x_stride_bytes, vl);
                                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2_m(v_active_lane_mask_b8, v_input_s8, vl);
                                vint16m2_t v_input_plus_offset_s16 = __riscv_vadd_vx_i16m2_m(v_active_lane_mask_b8, v_input_s16, s_input_offset_s16, vl);
                                v_acc_s32 = __riscv_vwmacc_vx_i32m4_m(v_active_lane_mask_b8, v_acc_s32, s_filter_val_s16, v_input_plus_offset_s16, vl);
                            }
                        }
                    }

                    // Requantize the accumulated values in a single function call.
                    vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
                        v_acc_s32,
                        scalar_multiplier,
                        effective_right_shift,
                        s_output_offset_s32,
                        s_output_activation_min_s32,
                        s_output_activation_max_s32,
                        vl);
                    
                    // Narrow result to int16 and then int8 with saturation
                    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
                    vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

                    // Store results vector (strided)
                    int8_t* output_strip_base_ptr = output_channel_base + current_out_x * output_w_stride;
                    __riscv_vsse8_v_i8m1(output_strip_base_ptr, output_x_stride_bytes, v_out_s8, vl);

                    // Advance output x pointer
                    current_out_x += vl;
                }
            }
        }
    }
}

void DepthwiseConvPerChannelRVV(const DepthwiseParams& params,
                           const int32_t* output_multiplier,
                           const int32_t* output_shift,
                           const RuntimeShape& input_shape,
                           const int8_t* input_data,
                           const RuntimeShape& filter_shape,
                           const int8_t* filter_data,
                           const RuntimeShape& bias_shape,
                           const int32_t* bias_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data)
{
    // Extract depthwise convolution parameters
    const int32_t input_offset = params.input_offset;
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const int pad_width = params.padding_values.width;
    const int pad_height = params.padding_values.height;
    const int depth_multiplier = params.depth_multiplier;
    const int32_t output_offset = params.output_offset;
    const int32_t output_activation_min = params.quantized_activation_min;
    const int32_t output_activation_max = params.quantized_activation_max;

    // Extract shape dimensions
    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
    const int input_batches = input_shape.Dims(0);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    // Calculate tensor strides
    const int input_ch_stride = 1;
    const int input_w_stride = input_depth;
    const int input_h_stride = input_width * input_w_stride;
    const int input_b_stride = input_height * input_h_stride;
    const int filter_ch_stride = 1;
    const int filter_w_stride = output_depth;
    const int filter_h_stride = filter_width * filter_w_stride;
    const int output_ch_stride = 1;
    const int output_w_stride = output_depth;
    const int output_h_stride = output_width * output_w_stride;
    const int output_b_stride = output_height * output_h_stride;

    // Prepare scalar constants
    const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    // Loop over batches
    for (int batch = 0; batch < input_batches; ++batch)
    {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;

        // Loop over output height
        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            const int in_y_origin = (out_y * stride_height) - pad_height;

            // Loop over input channels (depthwise)
            for (int in_channel = 0; in_channel < input_depth; ++in_channel)
            {
                // Loop over depth multiplier
                for (int m = 0; m < depth_multiplier; ++m)
                {
                    // Calculate the current output channel
                    const int output_channel = m + in_channel * depth_multiplier;

                    // Get per-channel requantization parameters
                    const int32_t scalar_multiplier = output_multiplier[output_channel];
                    const int32_t scalar_shift = output_shift[output_channel];
                    const int effective_right_shift = 31 - scalar_shift;

                    // Get bias value for this output channel
                    const int32_t bias_val = bias_data ? bias_data[output_channel] : 0;

                    // Calculate output pointer and stride for this channel row
                    int8_t* output_channel_row_base = output_batch_base + out_y * output_h_stride + output_channel * output_ch_stride;
                    const ptrdiff_t output_x_stride_bytes = output_w_stride * sizeof(int8_t);

                    // Process output width in vector chunks
                    size_t current_out_x = 0;
                    while (current_out_x < static_cast<size_t>(output_width))
                    {
                        // Set vector length for this iteration
                        size_t vl = __riscv_vsetvl_e32m4(output_width - current_out_x);

                        // Initialize accumulator vector with bias
                        vint32m4_t v_acc_s32 = bias_data ? __riscv_vmv_v_x_i32m4(bias_val, vl)
                                                         : __riscv_vmv_v_x_i32m4(0, vl);

                        // Calculate base input x coordinates for the vector lanes
                        vuint32m4_t v_idx = __riscv_vid_v_u32m4(vl);
                        vint32m4_t v_out_x = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vx_u32m4(v_idx, static_cast<uint32_t>(current_out_x), vl));
                        vint32m4_t v_in_x_origin_base = __riscv_vsub_vx_i32m4(__riscv_vmul_vx_i32m4(v_out_x, stride_width, vl), pad_width, vl);

                        // Loop over filter height
                        for (int filter_y = 0; filter_y < filter_height; ++filter_y)
                        {
                            const int in_y = in_y_origin + dilation_height_factor * filter_y;
                            if (in_y < 0 || in_y >= input_height) continue;

                            const int8_t* filter_y_base = filter_data + filter_y * filter_h_stride;

                            // Loop over filter width
                            for (int filter_x = 0; filter_x < filter_width; ++filter_x)
                            {
                                const int in_x_offset = dilation_width_factor * filter_x;
                                vint32m4_t v_in_x = __riscv_vadd_vx_i32m4(v_in_x_origin_base, in_x_offset, vl);

                                // Create mask for valid input coordinates
                                vbool8_t v_mask_ge_zero = __riscv_vmsge_vx_i32m4_b8(v_in_x, 0, vl);
                                vbool8_t v_mask_lt_width = __riscv_vmslt_vx_i32m4_b8(v_in_x, input_width, vl);
                                vbool8_t v_active_lane_mask_b8 = __riscv_vmand_mm_b8(v_mask_ge_zero, v_mask_lt_width, vl);

                                // Optimization: skip MAC if all lanes are masked off
                                if (__riscv_vfirst_m_b8(v_active_lane_mask_b8, vl) == -1) continue;

                                const int8_t* filter_ptr = filter_y_base + filter_x * filter_w_stride + output_channel * filter_ch_stride;
                                int16_t s_filter_val_s16 = static_cast<int16_t>(*filter_ptr);

                                int32_t base_in_x_for_vector0 = static_cast<int32_t>(current_out_x) * stride_width - pad_width + in_x_offset;
                                const int8_t* input_base_ptr =
                                  input_batch_base + in_y * input_h_stride + base_in_x_for_vector0 * input_w_stride + in_channel * input_ch_stride;
                                ptrdiff_t input_x_stride_bytes = static_cast<ptrdiff_t>(stride_width) * input_w_stride * sizeof(int8_t);

                                vint8m1_t v_input_s8 = __riscv_vlse8_v_i8m1_m(v_active_lane_mask_b8, input_base_ptr, input_x_stride_bytes, vl);
                                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2_m(v_active_lane_mask_b8, v_input_s8, vl);
                                vint16m2_t v_input_plus_offset_s16 = __riscv_vadd_vx_i16m2_m(v_active_lane_mask_b8, v_input_s16, s_input_offset_s16, vl);
                                v_acc_s32 = __riscv_vwmacc_vx_i32m4_m(v_active_lane_mask_b8, v_acc_s32, s_filter_val_s16, v_input_plus_offset_s16, vl);
                            }
                        }

                        // Requantize the accumulated values in a single function call.
                        vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
                            v_acc_s32,
                            scalar_multiplier,
                            effective_right_shift,
                            s_output_offset_s32,
                            s_output_activation_min_s32,
                            s_output_activation_max_s32,
                            vl);
                        
                        // Narrow result to int16 and then int8 with saturation
                        vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
                        vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

                        // Store results vector (strided)
                        int8_t* output_strip_base_ptr = output_channel_row_base + current_out_x * output_w_stride;
                        __riscv_vsse8_v_i8m1(output_strip_base_ptr, output_x_stride_bytes, v_out_s8, vl);

                        // Advance output x pointer
                        current_out_x += vl;
                    }
                }
            }
        }
    }
}