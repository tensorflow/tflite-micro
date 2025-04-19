#include <riscv_vector.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

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

    const int groups = input_depth / filter_input_depth;
    const int filters_per_group = output_depth / groups;

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

    const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    for (int batch = 0; batch < input_batches; ++batch) 
    {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;

        for (int out_y = 0; out_y < output_height; ++out_y) 
        {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            int8_t* output_row_base = output_batch_base + out_y * output_h_stride;

            for (int out_channel = 0; out_channel < output_depth; ++out_channel) 
            {
                const int group = out_channel / filters_per_group;
                const int group_start_input_channel = group * filter_input_depth;
                const int8_t* filter_oc_base = filter_data + out_channel * filter_o_stride;

                const int32_t scalar_multiplier = output_multiplier[out_channel];
                const int32_t scalar_shift = output_shift[out_channel];
                const int effective_right_shift = 31 - scalar_shift;

                const int32_t bias_val = bias_data ? bias_data[out_channel] : 0;

                int64_t rounding_val = (effective_right_shift > 0) ? (INT64_C(1) << (effective_right_shift - 1)) : 0;
                int32_t rounding_lo = (int32_t)rounding_val;
                int32_t rounding_hi = (int32_t)(rounding_val >> 32);

                int8_t* output_channel_base = output_row_base + out_channel * output_ch_stride;
                const ptrdiff_t output_x_stride_bytes = output_w_stride * sizeof(int8_t);

                size_t current_out_x = 0;
                while (current_out_x < (size_t)output_width) 
                {
                    size_t vl = __riscv_vsetvl_e32m4(output_width - current_out_x);

                    vint32m4_t v_acc_s32;
                    if (bias_data) 
                    {
                        v_acc_s32 = __riscv_vmv_v_x_i32m4(bias_val, vl);
                    } 
                    else 
                    {
                        v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vl);
                    }

                    vuint32m4_t v_idx = __riscv_vid_v_u32m4(vl);
                    vint32m4_t v_out_x = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vx_u32m4(v_idx, (uint32_t)current_out_x, vl));
                    vint32m4_t v_in_x_origin_base = __riscv_vsub_vx_i32m4(__riscv_vmul_vx_i32m4(v_out_x, stride_width, vl), pad_width, vl);

                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) 
                    {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        const bool is_y_inside_image = (in_y >= 0) && (in_y < input_height);

                        if (!is_y_inside_image)
                            continue;

                        const int8_t* filter_y_base = filter_oc_base + (filter_y * filter_h_stride);

                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) 
                        {
                            const int in_x_offset = dilation_width_factor * filter_x;
                            const int8_t* filter_patch_base = filter_y_base + (filter_x * filter_w_stride);

                            vint32m4_t v_in_x = __riscv_vadd_vx_i32m4(v_in_x_origin_base, in_x_offset, vl);

                            vbool8_t v_mask_ge_zero = __riscv_vmsge_vx_i32m4_b8(v_in_x, 0, vl);
                            vbool8_t v_mask_lt_width = __riscv_vmslt_vx_i32m4_b8(v_in_x, input_width, vl);
                            vbool8_t v_active_lane_mask_b8 = __riscv_vmand_mm_b8(v_mask_ge_zero, v_mask_lt_width, vl);

                            int32_t base_in_x_for_vector0 = (int32_t)current_out_x * stride_width - pad_width + in_x_offset;
                            const int8_t* input_base_for_y_x_patch = input_batch_base + (in_y * input_h_stride) + (base_in_x_for_vector0 * input_w_stride) +
                                                                     (group_start_input_channel * input_ch_stride);

                            ptrdiff_t input_x_stride_bytes = (ptrdiff_t)stride_width * input_w_stride * sizeof(int8_t);

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

                    vint32m4_t v_res32;

                    vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_acc_s32, scalar_multiplier, vl);
                    vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_acc_s32, scalar_multiplier, vl);

                    vuint32m4_t v_acc_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
                    vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_acc_lo_u, rounding_lo, vl);
                    vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding_lo, vl);
                    vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, rounding_hi, vl);
                    v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
                    vint32m4_t v_rounded_lo = __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_u);

                    if (effective_right_shift == 0) 
                    {
                        v_res32 = v_rounded_lo;
                    } 
                    else if (effective_right_shift > 0 && effective_right_shift < 32) 
                    {
                        vuint32m4_t v_lo_usrl = __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(v_rounded_lo), effective_right_shift, vl);
                        vint32m4_t v_hi_sll = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_right_shift, vl);
                        v_res32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4(v_lo_usrl, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_sll), vl));
                    } 
                    else 
                    {
                        int shift_hi = std::min(31, effective_right_shift - 32);
                        v_res32 = __riscv_vsra_vx_i32m4(v_rounded_hi, shift_hi, vl);
                    }

                    v_res32 = __riscv_vadd_vx_i32m4(v_res32, s_output_offset_s32, vl);

                    v_res32 = __riscv_vmax_vx_i32m4(v_res32, s_output_activation_min_s32, vl);
                    v_res32 = __riscv_vmin_vx_i32m4(v_res32, s_output_activation_max_s32, vl);

                    vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
                    vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

                    int8_t* output_strip_base_ptr = output_channel_base + current_out_x * output_w_stride;
                    __riscv_vsse8_v_i8m1(output_strip_base_ptr, output_x_stride_bytes, v_out_s8, vl);

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

    const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);

    const int input_batches = input_shape.Dims(0);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int input_depth = input_shape.Dims(3);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);

    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

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

    const int16_t s_input_offset_s16 = static_cast<int16_t>(input_offset);
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    for (int batch = 0; batch < input_batches; ++batch) 
    {
        const int8_t* input_batch_base = input_data + batch * input_b_stride;
        int8_t* output_batch_base = output_data + batch * output_b_stride;
        for (int out_y = 0; out_y < output_height; ++out_y) 
        {
            const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int in_channel = 0; in_channel < input_depth; ++in_channel) 
            {
                for (int m = 0; m < depth_multiplier; ++m) 
                {
                    const int output_channel = m + in_channel * depth_multiplier;
                    const int32_t scalar_multiplier = output_multiplier[output_channel];
                    const int32_t scalar_shift = output_shift[output_channel];
                    const int effective_right_shift = 31 - scalar_shift;

                    const int32_t bias_val = bias_data ? bias_data[output_channel] : 0;

                    int64_t rounding_val = (effective_right_shift > 0) ? (INT64_C(1) << (effective_right_shift - 1)) : 0;
                    int32_t rounding_lo = (int32_t)rounding_val;
                    int32_t rounding_hi = (int32_t)(rounding_val >> 32);

                    int8_t* output_channel_row_base = output_batch_base + out_y * output_h_stride + output_channel * output_ch_stride;

                    const ptrdiff_t output_x_stride_bytes = output_w_stride * sizeof(int8_t);

                    size_t current_out_x = 0;
                    while (current_out_x < (size_t)output_width) 
                    {

                        size_t vl = __riscv_vsetvl_e32m4(output_width - current_out_x);

                        vint32m4_t v_acc_s32;
                        if (bias_data) 
                        {
                            v_acc_s32 = __riscv_vmv_v_x_i32m4(bias_val, vl);
                        } 
                        else 
                        {
                            v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vl);
                        }

                        vuint32m4_t v_idx = __riscv_vid_v_u32m4(vl);
                        vint32m4_t v_out_x = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vadd_vx_u32m4(v_idx, (uint32_t)current_out_x, vl));
                        vint32m4_t v_in_x_origin_base = __riscv_vsub_vx_i32m4(__riscv_vmul_vx_i32m4(v_out_x, stride_width, vl), pad_width, vl);

                        for (int filter_y = 0; filter_y < filter_height; ++filter_y) 
                        {
                            const int in_y = in_y_origin + dilation_height_factor * filter_y;
                            const bool is_y_inside_image = (in_y >= 0) && (in_y < input_height);

                            if (!is_y_inside_image)
                                continue;

                            const int8_t* filter_y_base = filter_data + filter_y * filter_h_stride;

                            for (int filter_x = 0; filter_x < filter_width; ++filter_x) 
                            {
                                const int in_x_offset = dilation_width_factor * filter_x;
                                vint32m4_t v_in_x = __riscv_vadd_vx_i32m4(v_in_x_origin_base, in_x_offset, vl);

                                vbool8_t v_mask_ge_zero = __riscv_vmsge_vx_i32m4_b8(v_in_x, 0, vl);
                                vbool8_t v_mask_lt_width = __riscv_vmslt_vx_i32m4_b8(v_in_x, input_width, vl);
                                vbool8_t v_active_lane_mask_b8 = __riscv_vmand_mm_b8(v_mask_ge_zero, v_mask_lt_width, vl);

                                uint32_t first_mask_bit = __riscv_vfirst_m_b8(v_active_lane_mask_b8, vl);
                                if (first_mask_bit == (uint32_t)-1 && vl > 0)
                                    continue;

                                const int8_t* filter_ptr = filter_y_base + filter_x * filter_w_stride + output_channel * filter_ch_stride;
                                int8_t s_filter_val_s8 = *filter_ptr;
                                int16_t s_filter_val_s16 = static_cast<int16_t>(s_filter_val_s8);

                                int32_t base_in_x_for_vector0 = (int32_t)current_out_x * stride_width - pad_width + in_x_offset;
                                const int8_t* input_base_ptr =
                                  input_batch_base + in_y * input_h_stride + base_in_x_for_vector0 * input_w_stride + in_channel * input_ch_stride;

                                ptrdiff_t input_x_stride_bytes = (ptrdiff_t)stride_width * input_w_stride * sizeof(int8_t);

                                vint8m1_t v_input_s8 = __riscv_vlse8_v_i8m1_m(v_active_lane_mask_b8, input_base_ptr, input_x_stride_bytes, vl);
                                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2_m(v_active_lane_mask_b8, v_input_s8, vl);
                                vint16m2_t v_input_plus_offset_s16 = __riscv_vadd_vx_i16m2_m(v_active_lane_mask_b8, v_input_s16, s_input_offset_s16, vl);

                                v_acc_s32 = __riscv_vwmacc_vx_i32m4_m(v_active_lane_mask_b8, v_acc_s32, s_filter_val_s16, v_input_plus_offset_s16, vl);
                            }
                        }

                        vint32m4_t v_res32;

                        vint32m4_t v_prod_lo = __riscv_vmul_vx_i32m4(v_acc_s32, scalar_multiplier, vl);
                        vint32m4_t v_prod_hi = __riscv_vmulh_vx_i32m4(v_acc_s32, scalar_multiplier, vl);

                        vuint32m4_t v_acc_lo_u = __riscv_vreinterpret_v_i32m4_u32m4(v_prod_lo);
                        vuint32m4_t v_sum_lo_u = __riscv_vadd_vx_u32m4(v_acc_lo_u, rounding_lo, vl);
                        vbool8_t v_carry = __riscv_vmsltu_vx_u32m4_b8(v_sum_lo_u, rounding_lo, vl);
                        vint32m4_t v_rounded_hi = __riscv_vadd_vx_i32m4(v_prod_hi, rounding_hi, vl);
                        v_rounded_hi = __riscv_vadd_vx_i32m4_m(v_carry, v_rounded_hi, 1, vl);
                        vint32m4_t v_rounded_lo = __riscv_vreinterpret_v_u32m4_i32m4(v_sum_lo_u);

                        if (effective_right_shift == 0) 
                        {
                            v_res32 = v_rounded_lo;
                        } 
                        else if (effective_right_shift > 0 && effective_right_shift < 32) 
                        {
                            vuint32m4_t v_lo_usrl = __riscv_vsrl_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(v_rounded_lo), effective_right_shift, vl);
                            vint32m4_t v_hi_sll = __riscv_vsll_vx_i32m4(v_rounded_hi, 32 - effective_right_shift, vl);
                            v_res32 = __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vor_vv_u32m4(v_lo_usrl, __riscv_vreinterpret_v_i32m4_u32m4(v_hi_sll), vl));
                        } 
                        else 
                        {
                            int shift_hi = std::min(31, effective_right_shift - 32);
                            v_res32 = __riscv_vsra_vx_i32m4(v_rounded_hi, shift_hi, vl);
                        }

                        v_res32 = __riscv_vadd_vx_i32m4(v_res32, s_output_offset_s32, vl);

                        v_res32 = __riscv_vmax_vx_i32m4(v_res32, s_output_activation_min_s32, vl);
                        v_res32 = __riscv_vmin_vx_i32m4(v_res32, s_output_activation_max_s32, vl);

                        vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
                        vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);

                        int8_t* output_strip_base_ptr = output_channel_row_base + current_out_x * output_w_stride;
                        __riscv_vsse8_v_i8m1(output_strip_base_ptr, output_x_stride_bytes, v_out_s8, vl);

                        current_out_x += vl;
                    }
                }
            }
        }
    }
}