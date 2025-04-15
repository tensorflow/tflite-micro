#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <limits> 

#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"

using namespace tflite;

void ConvPerChannelRVV(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
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

    const int filter_output_depth = filter_shape.Dims(0);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int filter_input_depth = filter_shape.Dims(3);

    const int output_batches = output_shape.Dims(0);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);
    const int output_depth_dim = output_shape.Dims(3);

    const int batches = std::min(input_batches, output_batches);
    const int output_depth = std::min(filter_output_depth, output_depth_dim);

    const int groups = input_depth / filter_input_depth;
    const int filters_per_group = output_depth / groups;

    const int16_t input_offset_s16 = static_cast<int16_t>(input_offset);

    const int input_ch_stride = 1;
    const int input_w_stride = input_depth * input_ch_stride;
    const int input_h_stride = input_width * input_w_stride;
    const int input_b_stride = input_height * input_h_stride;

    const int filter_ch_stride = 1;
    const int filter_w_stride = filter_input_depth * filter_ch_stride;
    const int filter_h_stride = filter_width * filter_w_stride;
    const int filter_o_stride = filter_height * filter_h_stride;

    const int output_ch_stride = 1;
    const int output_w_stride = output_depth * output_ch_stride;
    const int output_h_stride = output_width * output_w_stride;
    const int output_b_stride = output_height * output_h_stride;

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* input_batch_base = input_data + batch * input_b_stride;
      int8_t* output_batch_base = output_data + batch * output_b_stride;

      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;

        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;

          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            const int group = out_channel / filters_per_group;
            const int group_start_input_channel = group * filter_input_depth;
            int32_t acc = 0;

            const int8_t* filter_oc_base = filter_data + out_channel * filter_o_stride;

            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const int8_t* filter_y_base = filter_oc_base + filter_y * filter_h_stride;

              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int8_t* filter_x_base = filter_y_base + filter_x * filter_w_stride;

                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                if (!is_point_inside_image) {
                  continue;
                }

                const int input_offset_addr = (in_y * input_h_stride) + (in_x * input_w_stride) + (group_start_input_channel * input_ch_stride);
                const int8_t* input_ptr = input_batch_base + input_offset_addr;
                const int8_t* filter_ptr = filter_x_base;

                size_t channels_remaining = filter_input_depth;
                int32_t patch_acc = 0;

                if (channels_remaining > 0) {
                    size_t vlmax_for_acc = __riscv_vsetvlmax_e32m4();
                    vint32m4_t v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vlmax_for_acc);

                    while (channels_remaining > 0) {
                        // Use LMUL=1 for 8-bit loads
                        size_t current_vl = __riscv_vsetvl_e8m1(channels_remaining);

                        // Load 8-bit data into m1 vectors
                        vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_ptr, current_vl);
                        vint8m1_t v_filter_s8 = __riscv_vle8_v_i8m1(filter_ptr, current_vl);

                        // Widen 8m1 -> 16m2
                        vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, current_vl);
                        vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, current_vl);

                        // Perform add on 16m2 vectors
                        v_input_s16 = __riscv_vadd_vx_i16m2(v_input_s16, input_offset_s16, current_vl);

                        // Widening multiply-accumulate: 16m2 * 16m2 + 32m4 -> 32m4
                        // Pass current_vl, the number of elements processed in this iteration
                        v_acc_s32 = __riscv_vwmacc_vv_i32m4(v_acc_s32, v_filter_s16, v_input_s16, current_vl);

                        input_ptr += current_vl;
                        filter_ptr += current_vl;
                        channels_remaining -= current_vl;
                    }

                    // Reduce the final 32m4 accumulator
                    size_t vl_for_reduce = __riscv_vsetvl_e32m4(filter_input_depth); // Set VL for the reduction source type
                    vint32m1_t v_zero_reduction = __riscv_vmv_s_x_i32m1(0, 1);
                    vint32m1_t v_sum_reduction = __riscv_vredsum_vs_i32m4_i32m1(
                                                    v_acc_s32,
                                                    v_zero_reduction,
                                                    vl_for_reduce); // Use the VL corresponding to the accumulator length

                    patch_acc = __riscv_vmv_x_s_i32m1_i32(v_sum_reduction);
                }
                acc += patch_acc;
              }
            }

            if (bias_data) {
              acc += bias_data[out_channel];
            }

            const int32_t current_multiplier = output_multiplier[out_channel];
            const int32_t current_shift = output_shift[out_channel];
            const int64_t total_shift = 31 - current_shift;
            const int64_t round_val = (total_shift > 0) ? (static_cast<int64_t>(1) << (total_shift - 1)) : 0LL;
            int64_t result64 = static_cast<int64_t>(acc) * static_cast<int64_t>(current_multiplier);
            result64 += round_val;
            result64 = result64 >> total_shift;
            result64 = std::max(result64, static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
            result64 = std::min(result64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
            acc = static_cast<int32_t>(result64);

            acc += output_offset;

            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);

            const int output_offset_addr = (out_y * output_h_stride) + (out_x * output_w_stride) + (out_channel * output_ch_stride);
            output_batch_base[output_offset_addr] = static_cast<int8_t>(acc);

          }
        }
      }
    }
}