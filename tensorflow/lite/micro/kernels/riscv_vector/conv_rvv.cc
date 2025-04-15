#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <limits> 

#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

using namespace tflite;

void ConvPerChannelRVV(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data)
{
    MicroPrintf("[PEANUT MICROSYSTEMS] ConvPerChannelRVV");

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

    // Extract dimensions from input, filter, and output shapes
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

    // Determine the actual number of batches and output channels to process
    const int batches = std::min(input_batches, output_batches);
    const int output_depth = std::min(filter_output_depth, output_depth_dim);

    // Calculate group information for grouped/depthwise convolutions
    const int groups = input_depth / filter_input_depth;
    const int filters_per_group = output_depth / groups;

    // Prepare input offset as int16_t for vector operations
    const int16_t input_offset_s16 = static_cast<int16_t>(input_offset);

    // Calculate memory strides for navigating input, filter, and output tensors
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

    for (int batch = 0; batch < batches; ++batch) 
    {
      // Get base pointers for the current batch's input and output data
      const int8_t* input_batch_base = input_data + batch * input_b_stride;
      int8_t* output_batch_base = output_data + batch * output_b_stride;

      for (int out_y = 0; out_y < output_height; ++out_y) 
      {
        // Calculate the starting row index in the input tensor corresponding to the current output row
        const int in_y_origin = (out_y * stride_height) - pad_height;

        for (int out_x = 0; out_x < output_width; ++out_x) 
        {
          // Calculate the starting column index in the input tensor corresponding to the current output column
          const int in_x_origin = (out_x * stride_width) - pad_width;

          for (int out_channel = 0; out_channel < output_depth; ++out_channel) 
          {
            // Determine the group index and starting input channel for the current output channel
            const int group = out_channel / filters_per_group;
            const int group_start_input_channel = group * filter_input_depth;

            // Initialize the accumulator for the current output pixel and channel
            int32_t acc = 0;

            // Get the base pointer for the filter data corresponding to the current output channel
            const int8_t* filter_oc_base = filter_data + out_channel * filter_o_stride;

            for (int filter_y = 0; filter_y < filter_height; ++filter_y) 
            {
              // Calculate the corresponding row index in the input tensor
              const int in_y = in_y_origin + dilation_height_factor * filter_y;

              // Get the base pointer for the current filter row
              const int8_t* filter_y_base = filter_oc_base + filter_y * filter_h_stride;

              for (int filter_x = 0; filter_x < filter_width; ++filter_x) 
              {
                // Calculate the corresponding column index in the input tensor
                const int in_x = in_x_origin + dilation_width_factor * filter_x;

                // Get the base pointer for the current filter column
                const int8_t* filter_x_base = filter_y_base + filter_x * filter_w_stride;

                // Check if the calculated input patch position is within the input tensor boundaries
                const bool is_point_inside_image =
                    (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height);

                // Skip computation if the current filter patch position is outside the input boundaries
                if (!is_point_inside_image) 
                  continue;

                // Calculate the memory offset to the start of the relevant input data patch
                const int input_offset_addr = (in_y * input_h_stride) + (in_x * input_w_stride) + (group_start_input_channel * input_ch_stride);
                // Get pointers to the start of the input patch and corresponding filter data
                const int8_t* input_ptr = input_batch_base + input_offset_addr;
                const int8_t* filter_ptr = filter_x_base;

                // Initialize variables for the vector processing loop over input channels for this patch
                size_t channels_remaining = filter_input_depth;
                int32_t patch_acc = 0;

                // Perform vector MAC operation if there are channels to process for this patch
                if (channels_remaining > 0)
                {
                    // Initialize a 32-bit vector accumulator (m4) to zeros
                    size_t vlmax_for_acc = __riscv_vsetvlmax_e32m4();
                    vint32m4_t v_acc_s32 = __riscv_vmv_v_x_i32m4(0, vlmax_for_acc);

                    // Process input channels in vector chunks until all are done
                    while (channels_remaining > 0)
                    {
                        // Set the vector length for the current iteration
                        size_t current_vl = __riscv_vsetvl_e8m1(channels_remaining);

                        // Load 8-bit input and filter data chunks into m1 vectors
                        vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_ptr, current_vl);
                        vint8m1_t v_filter_s8 = __riscv_vle8_v_i8m1(filter_ptr, current_vl);

                        // Widen 8-bit vectors (m1) to 16-bit vectors (m2)
                        vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, current_vl);
                        vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, current_vl);

                        // Add the input offset to the widened 16-bit input vector
                        v_input_s16 = __riscv_vadd_vx_i16m2(v_input_s16, input_offset_s16, current_vl);

                        // Perform widening multiply-accumulate: 16m2 * 16m2 -> 32m4, accumulating into v_acc_s32
                        v_acc_s32 = __riscv_vwmacc_vv_i32m4(v_acc_s32, v_filter_s16, v_input_s16, current_vl);

                        // Advance input and filter pointers and decrement remaining channel count
                        input_ptr += current_vl;
                        filter_ptr += current_vl;
                        channels_remaining -= current_vl;
                    }

                    // Reduce the final 32-bit vector accumulator to a scalar sum
                    size_t vl_for_reduce = __riscv_vsetvl_e32m4(filter_input_depth);
                    vint32m1_t v_zero_reduction = __riscv_vmv_s_x_i32m1(0, 1);
                    vint32m1_t v_sum_reduction = __riscv_vredsum_vs_i32m4_i32m1(
                                                    v_acc_s32,
                                                    v_zero_reduction,
                                                    vl_for_reduce);

                    // Extract the scalar reduction result into patch_acc
                    patch_acc = __riscv_vmv_x_s_i32m1_i32(v_sum_reduction);
                }

                // Accumulate the result from the processed patch into the overall accumulator
                acc += patch_acc;
              }
            }

            // Add bias value
            if (bias_data) {
              acc += bias_data[out_channel];
            }

            // Apply per-channel requantization to the accumulated value
            const int32_t current_multiplier = output_multiplier[out_channel];
            const int32_t current_shift = output_shift[out_channel];
            const int64_t total_shift = 31 - current_shift;
            const int64_t round_val = (total_shift > 0) ? (static_cast<int64_t>(1) << (total_shift - 1)) : 0LL;
            int64_t result64 = static_cast<int64_t>(acc) * static_cast<int64_t>(current_multiplier);
            result64 += round_val; // Add rounding value
            result64 = result64 >> total_shift; // Perform the shift
            result64 = std::max(result64, static_cast<int64_t>(std::numeric_limits<int32_t>::min()));
            result64 = std::min(result64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
            acc = static_cast<int32_t>(result64);

            // Add the output offset to the requantized value
            acc += output_offset;

            // Clamp the result to the final activation range
            acc = std::max(acc, output_activation_min);
            acc = std::min(acc, output_activation_max);

            // Calculate the memory offset for the output pixel and store the final 8-bit result
            const int output_offset_addr = (out_y * output_h_stride) + (out_x * output_w_stride) + (out_channel * output_ch_stride);
            output_batch_base[output_offset_addr] = static_cast<int8_t>(acc);
          }
        }
      }
    }
}