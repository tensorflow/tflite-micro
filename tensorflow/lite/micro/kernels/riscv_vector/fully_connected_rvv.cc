#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

using namespace tflite;

void FullyConnectedPerChannelRVV(const FullyConnectedParams& params,
                            const int32_t* output_multiplier,
                            const int* output_shift,
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
    const int32_t output_offset = params.output_offset;
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
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    // Loop over batches
    for (int b = 0; b < batches; ++b)
    {
        // Set base pointers for the current batch
        const int8_t* input_batch_ptr = input_data + b * accum_depth;
        int8_t* output_batch_ptr = output_data + b * output_depth;

        // Loop over output channels (rows of the weight matrix)
        for (int out_c = 0; out_c < output_depth; ++out_c) {
            // Set filter pointer and get bias for the current output channel
            const int8_t* filter_row_ptr = filter_data + out_c * accum_depth;
            const int32_t bias_val = bias_data ? bias_data[out_c] : 0;

            // Initialize vector accumulator to zero
            size_t initial_vl_for_acc_init = __riscv_vsetvlmax_e16m2();
            vint32m4_t v_acc_s32m4 = __riscv_vmv_v_x_i32m4(0, initial_vl_for_acc_init);

            // Initialize scalar accumulator with bias value
            int32_t s_acc_s32 = bias_val;

            // Loop over accumulation depth (dot product length) in vector
            // chunks
            size_t current_d = 0;
            while (current_d < static_cast<size_t>(accum_depth))
            {
                // Set vector length for the current chunk
                size_t vl = __riscv_vsetvl_e16m2(accum_depth - current_d);

                // Load input vector chunk, widen to i16, and add input offset
                vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_batch_ptr + current_d, vl);
                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
                vint16m2_t v_input_plus_offset_s16 = __riscv_vadd_vx_i16m2(v_input_s16, s_input_offset_s16, vl);

                // Load filter vector chunk and widen to i16
                vint8m1_t v_filter_s8 = __riscv_vle8_v_i8m1(filter_row_ptr + current_d, vl);
                vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);

                // Perform widening multiply-accumulate
                v_acc_s32m4 = __riscv_vwmacc_vv_i32m4(v_acc_s32m4, v_input_plus_offset_s16, v_filter_s16, vl);

                // Advance pointer for the next chunk
                current_d += vl;
            }

            // Reduce the final vector accumulator to a scalar sum
            size_t final_vl = __riscv_vsetvl_e32m4(accum_depth > 0 ? 1 : 0);
            if (accum_depth > 0)
            {
                // Set VL for reduction based on accumulated depth
                final_vl = __riscv_vsetvl_e32m4(accum_depth);

                // Initialize reduction target vector register to zero
                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

                // Perform reduction sum
                vint32m1_t v_reduced_sum_s32m1 = __riscv_vredsum_vs_i32m4_i32m1(v_acc_s32m4, v_zero, final_vl);

                // Extract scalar sum and add to the bias-initialized scalar accumulator
                s_acc_s32 += __riscv_vmv_x_s_i32m1_i32(v_reduced_sum_s32m1);
            }

            // Apply per-channel requantization (scalar multiplication and shift)
            int32_t s_requantized_acc_s32 = MultiplyByQuantizedMultiplier(s_acc_s32, output_multiplier[out_c], output_shift[out_c]);

            // Add output offset to the requantized value
            s_requantized_acc_s32 += s_output_offset_s32;

            // Clamp the result to the activation range
            s_requantized_acc_s32 = std::max(s_requantized_acc_s32, s_output_activation_min_s32);
            s_requantized_acc_s32 = std::min(s_requantized_acc_s32, s_output_activation_max_s32);

            // Store the final int8 result
            output_batch_ptr[out_c] = static_cast<int8_t>(s_requantized_acc_s32);
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
    // Extract quantization parameters (scalar values for the whole layer)
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
    const int32_t s_output_offset_s32 = output_offset;
    const int32_t s_output_activation_min_s32 = output_activation_min;
    const int32_t s_output_activation_max_s32 = output_activation_max;

    // Loop over batches
    for (int b = 0; b < batches; ++b) 
    {
        // Set base pointers for the current batch
        const int8_t* input_batch_ptr = input_data + b * accum_depth;
        int8_t* output_batch_ptr = output_data + b * output_depth;

        // Loop over output channels (rows of the weight matrix)
        for (int out_c = 0; out_c < output_depth; ++out_c)
        {
            // Set filter pointer and get bias for the current output channel
            const int8_t* filter_row_ptr = filter_data + out_c * accum_depth;
            // Bias is int32_t for non-per-channel int8 quantization
            const int32_t bias_val = bias_data ? bias_data[out_c] : 0;

            // Initialize vector accumulator to zero
            // Use vlmax corresponding to operand type (e16m2) to determine acc size
            size_t initial_vl_for_acc_init = __riscv_vsetvlmax_e16m2();
            vint32m4_t v_acc_s32m4 = __riscv_vmv_v_x_i32m4(0, initial_vl_for_acc_init);

            // Initialize scalar accumulator with bias value
            int32_t s_acc_s32 = bias_val;

            // Loop over accumulation depth (dot product length) in vector chunks
            size_t current_d = 0;
            while (current_d < static_cast<size_t>(accum_depth))
            {
                // Set vector length for the current chunk
                size_t vl = __riscv_vsetvl_e16m2(accum_depth - current_d);

                // Load input vector chunk, widen to i16, and add input offset
                vint8m1_t v_input_s8 = __riscv_vle8_v_i8m1(input_batch_ptr + current_d, vl);
                vint16m2_t v_input_s16 = __riscv_vsext_vf2_i16m2(v_input_s8, vl);
                vint16m2_t v_input_plus_offset_s16 = __riscv_vadd_vx_i16m2(v_input_s16, s_input_offset_s16, vl);

                // Load filter vector chunk, widen to i16, and add filter offset
                vint8m1_t v_filter_s8 = __riscv_vle8_v_i8m1(filter_row_ptr + current_d, vl);
                vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);
                vint16m2_t v_filter_plus_offset_s16 = __riscv_vadd_vx_i16m2(v_filter_s16, s_filter_offset_s16, vl);

                // Perform widening multiply-accumulate
                v_acc_s32m4 = __riscv_vwmacc_vv_i32m4(v_acc_s32m4, v_input_plus_offset_s16, v_filter_plus_offset_s16, vl);

                // Advance pointer for the next chunk
                current_d += vl;
            }

            // Reduce the final vector accumulator to a scalar sum
            size_t final_vl = __riscv_vsetvl_e32m4(accum_depth > 0 ? 1 : 0);
            if (accum_depth > 0)
            {
                // Set VL for reduction based on accumulated depth
                final_vl = __riscv_vsetvl_e32m4(accum_depth);

                // Initialize reduction target vector register to zero
                vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, 1);

                // Perform reduction sum
                vint32m1_t v_reduced_sum_s32m1 = __riscv_vredsum_vs_i32m4_i32m1(v_acc_s32m4, v_zero, final_vl);

                // Extract scalar sum and add to the bias-initialized scalar accumulator
                s_acc_s32 += __riscv_vmv_x_s_i32m1_i32(v_reduced_sum_s32m1);
            }

            // Apply uniform requantization (scalar multiplication and shift)
            int32_t s_requantized_acc_s32 = MultiplyByQuantizedMultiplier(s_acc_s32, output_multiplier, output_shift);

            // Add output offset to the requantized value
            s_requantized_acc_s32 += s_output_offset_s32;

            // Clamp the result to the activation range
            s_requantized_acc_s32 = std::max(s_requantized_acc_s32, s_output_activation_min_s32);
            s_requantized_acc_s32 = std::min(s_requantized_acc_s32, s_output_activation_max_s32);

            // Store the final int8 result (using batch offset)
            output_batch_ptr[out_c] = static_cast<int8_t>(s_requantized_acc_s32);
        }
    }
}
