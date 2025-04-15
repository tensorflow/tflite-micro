#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <limits> 

#include <riscv_vector.h>

// TFLite Micro reference
// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 32 bits accumulator.
                // In the nudging process during model quantization, we force
                // real value of 0.0 be represented by a quantized value. This
                // guarantees that the input_offset is a int8_t, even though
                // it is represented using int32_t. int32_t += int8_t *
                // (int8_t - int8_t) so the highest value we can get from each
                // accumulation is [-127, 127] * ([-128, 127] -
                // [-128, 127]), which is [-32512, 32512]. log2(32512)
                // = 14.98, which means we can accumulate at least 2^16
                // multiplications without overflow. The accumulator is
                // applied to a filter so the accumulation logic will hold as
                // long as the filter size (filter_y * filter_x * in_channel)
                // does not exceed 2^16, which is the case in all the models
                // we have seen so far.
                // TODO(b/174275578): Add a check to make sure the
                // accumulator depth is smaller than 2^16.
                acc += filter_val * (input_val + input_offset);
              }
            }
          }

          if (bias_data) {
            acc += bias_data[out_channel];
          }
          acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::max(acc, output_activation_min);
          acc = std::min(acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int8_t>(acc);
        }
      }
    }
  }
}

__attribute__((hot))
void convolution_hwc_ohwi_rvv(
    const int8_t* input_data,
    const uint16_t input_height,
    const uint16_t input_width,
    const uint16_t input_channels,
    const int32_t input_offset,
    const int8_t* filter_data,
    const uint16_t filter_height,
    const uint16_t filter_width,
    const int32_t* bias_data,
    int8_t* output_data,
    const uint16_t output_height,
    const uint16_t output_width,
    const uint16_t output_channels,
    const int32_t output_offset,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const uint16_t stride_height,
    const uint16_t stride_width,
    const uint16_t pad_height,
    const uint16_t pad_width,
    const int32_t output_activation_min,
    const int32_t output_activation_max,
    int dilation_height_factor,
    int dilation_width_factor
)
{
    assert(input_data != nullptr);
    assert(filter_data != nullptr);

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

    assert(input_offset >= -128 && input_offset <= 127);
    assert(output_offset >= -128 && output_offset <= 127);


    const size_t input_row_stride = (size_t)input_width * input_channels;
    const size_t output_row_stride = (size_t)output_width * output_channels;
    const size_t filter_kernel_plane_size = (size_t)filter_height * filter_width * input_channels;

    const int32_t output_activation_min_i32 = output_activation_min;
    const int32_t output_activation_max_i32 = output_activation_max;

    const unsigned int default_vxrm = __RISCV_VXRM_RNU;

    for (int out_c = 0; out_c < output_channels; ++out_c)
    {
        const int32_t current_bias = (bias_data != nullptr) ? bias_data[out_c] : 0;
        const int32_t current_output_multiplier = output_multiplier[out_c];
        const int32_t current_output_shift = output_shift[out_c];

        const int32_t left_shift = std::max((int32_t)0, current_output_shift);
        const int32_t right_shift = std::max((int32_t)0, -current_output_shift);

        const int32_t rounding_offset = (right_shift > 0) ? (1 << (right_shift - 1)) : 0;

        const int32_t add_rounding_limit = INT32_MAX - rounding_offset;
        const int32_t add_offset_limit_pos = INT32_MAX - output_offset;
        const int32_t add_offset_limit_neg = INT32_MIN - output_offset;

        const int32_t left_shift_limit_pos = (left_shift < 31) ? (INT32_MAX >> left_shift) : 0;
        const int32_t left_shift_limit_neg = (left_shift < 31) ? (INT32_MIN >> left_shift) : -1;

        for (int out_y = 0; out_y < output_height; ++out_y)
        {
            const int in_y_origin = (out_y * stride_height) - pad_height;

            size_t current_out_x = 0;
            while (current_out_x < output_width)
            {
                const size_t vl = __riscv_vsetvl_e32m8(output_width - current_out_x);

                vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vl);

                for (int k_y = 0; k_y < filter_height; ++k_y)
                {
                    const int in_y = in_y_origin + k_y * dilation_height_factor;

                    if (in_y < 0 || in_y >= input_height) continue;

                    for (int k_x = 0; k_x < filter_width; ++k_x)
                    {
                        vuint32m8_t v_lane_indices = __riscv_vid_v_u32m8(vl);
                        vuint32m8_t v_out_x_indices = __riscv_vadd_vx_u32m8(v_lane_indices, current_out_x, vl);
                        vuint32m8_t v_in_x_base = __riscv_vmul_vx_u32m8(v_out_x_indices, stride_width, vl);
                        const int32_t in_x_origin_for_k = (int32_t)(k_x * dilation_width_factor) - pad_width;
                        vint32m8_t v_in_x_i32 = __riscv_vadd_vx_i32m8(__riscv_vreinterpret_v_u32m8_i32m8(v_in_x_base), in_x_origin_for_k, vl);

                        vbool4_t v_mask_x_ge_0 = __riscv_vmsge_vx_i32m8_b4(v_in_x_i32, 0, vl);
                        vbool4_t v_mask_x_lt_w = __riscv_vmslt_vx_i32m8_b4(v_in_x_i32, input_width, vl);
                        vbool4_t v_mask_valid_x = __riscv_vmand_mm_b4(v_mask_x_ge_0, v_mask_x_lt_w, vl);

                        vuint32m8_t v_in_x_u32 = __riscv_vreinterpret_v_i32m8_u32m8(v_in_x_i32);
                        vuint32m8_t v_in_x_ch_offset = __riscv_vmul_vx_u32m8(v_in_x_u32, input_channels, vl);

                        const int8_t* filter_ptr_base = filter_data +
                            (size_t)out_c * filter_kernel_plane_size +
                            (size_t)k_y * filter_width * input_channels +
                            (size_t)k_x * input_channels;

                        for (int in_c = 0; in_c < input_channels; ++in_c)
                        {
                            const int8_t filter_val = filter_ptr_base[in_c];

                            if (filter_val == 0)
                                continue;

                            uint32_t base_offset_for_row_ch = (uint32_t)in_y * input_row_stride + in_c;
                            vuint32m8_t v_byte_offset_u32 = __riscv_vadd_vx_u32m8(v_in_x_ch_offset, base_offset_for_row_ch, vl);

                            vint8m2_t v_loaded_input_i8 = __riscv_vloxei32_v_i8m2_m(
                                                            v_mask_valid_x,
                                                            input_data,
                                                            v_byte_offset_u32,
                                                            vl);

                            vint16m4_t v_input_i16 = __riscv_vsext_vf2_i16m4(v_loaded_input_i8, vl);

                            vint16m4_t v_input_offset_i16 = __riscv_vmv_v_x_i16m4((int16_t)input_offset, vl);
                            vint16m4_t v_input_plus_offset_all = __riscv_vadd_vv_i16m4(v_input_i16, v_input_offset_i16, vl);

                            v_acc = __riscv_vwmacc_vx_i32m8(v_acc,
                                                            filter_val,
                                                            v_input_plus_offset_all,
                                                            vl);
                        }
                    }
                }

                if (bias_data != nullptr) {
                    v_acc = __riscv_vadd_vx_i32m8(v_acc, current_bias, vl);
                }                 

                 vint32m8_t v_requant_stage1 = __riscv_vmulh_vx_i32m8(v_acc, current_output_multiplier, vl);

                if (right_shift > 0)
                {
                    vbool4_t v_mask_add_round_ovf = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, add_rounding_limit, vl);
                    vint32m8_t v_add_round_sat = __riscv_vmerge_vxm_i32m8(v_requant_stage1, INT32_MAX, v_mask_add_round_ovf, vl);

                    vint32m8_t v_added_round = __riscv_vadd_vx_i32m8(v_add_round_sat, rounding_offset, vl);

                    v_requant_stage1 = __riscv_vsra_vx_i32m8(v_added_round, right_shift, vl);
                 }

                if (left_shift > 0)
                {
                    vbool4_t v_mask_lshift_ovf_pos = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, left_shift_limit_pos, vl);
                    vbool4_t v_mask_lshift_ovf_neg = __riscv_vmslt_vx_i32m8_b4(v_requant_stage1, left_shift_limit_neg, vl);

                    vint32m8_t v_shifted = __riscv_vsll_vx_i32m8(v_requant_stage1, left_shift, vl);

                    v_shifted = __riscv_vmerge_vxm_i32m8(v_shifted, INT32_MAX, v_mask_lshift_ovf_pos, vl);
                    v_shifted = __riscv_vmerge_vxm_i32m8(v_shifted, INT32_MIN, v_mask_lshift_ovf_neg, vl);
                    v_requant_stage1 = v_shifted;
                }

                vbool4_t v_mask_add_offset_ovf_pos = __riscv_vmsgt_vx_i32m8_b4(v_requant_stage1, add_offset_limit_pos, vl);
                vbool4_t v_mask_add_offset_ovf_neg = __riscv_vmslt_vx_i32m8_b4(v_requant_stage1, add_offset_limit_neg, vl);

                vint32m8_t v_requant_stage2 = __riscv_vadd_vx_i32m8(v_requant_stage1, output_offset, vl);

                v_requant_stage2 = __riscv_vmerge_vxm_i32m8(v_requant_stage2, INT32_MAX, v_mask_add_offset_ovf_pos, vl);
                v_requant_stage2 = __riscv_vmerge_vxm_i32m8(v_requant_stage2, INT32_MIN, v_mask_add_offset_ovf_neg, vl);

                vint32m8_t v_clamped_i32 = __riscv_vmax_vx_i32m8(v_requant_stage2, output_activation_min_i32, vl);
                v_clamped_i32 = __riscv_vmin_vx_i32m8(v_clamped_i32, output_activation_max_i32, vl);

                vint16m4_t v_narrowed_i16 = __riscv_vnsra_wx_i16m4(v_clamped_i32, 0, vl);

                __riscv_csrw(CSR_VXRM, __RISCV_VXRM_RDN);

                vint8m2_t v_output_i8 = __riscv_vnclip_wx_i8m2(v_narrowed_i16, 0, default_vxrm, vl);

                int8_t* output_base_ptr = output_data +
                                            (size_t)out_y * output_row_stride +
                                            current_out_x * output_channels +
                                            out_c;

                ptrdiff_t byte_stride = (ptrdiff_t)output_channels * sizeof(int8_t);

                __riscv_vsse8_v_i8m2(output_base_ptr,
                                     byte_stride,
                                     v_output_i8,
                                     vl);

                current_out_x += vl;
            }
        }
    }
}