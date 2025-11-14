#include <riscv_vector.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

#include "requantize_rvv.h"

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
    for (int b = 0; b < batches; ++b)
    {
        const int8_t* input_batch_ptr = input_data + b * accum_depth;
        int8_t* output_batch_ptr = output_data + b * output_depth;

        // Vectorized loop over output channels
        size_t current_out_c = 0;
        while (current_out_c < static_cast<size_t>(output_depth))
        {
            // Set vector length for this iteration
            size_t vl = __riscv_vsetvl_e32m4(output_depth - current_out_c);

            // Initialize accumulator vector with biases
            vint32m4_t v_acc_s32 = bias_data
                ? __riscv_vle32_v_i32m4(bias_data + current_out_c, vl)
                : __riscv_vmv_v_x_i32m4(0, vl);

            // Main MAC loop to compute dot products
            for (int d = 0; d < accum_depth; ++d)
            {
                int16_t s_input_val_s16 = static_cast<int16_t>(input_batch_ptr[d]) + s_input_offset_s16;
                const int8_t* filter_col_ptr = filter_data + d + current_out_c * accum_depth;
                ptrdiff_t filter_stride = accum_depth * sizeof(int8_t);
                vint8m1_t v_filter_s8 = __riscv_vlse8_v_i8m1(filter_col_ptr, filter_stride, vl);
                vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);
                v_acc_s32 = __riscv_vwmacc_vx_i32m4(v_acc_s32, s_input_val_s16, v_filter_s16, vl);
            }

            // Load per-channel requantization parameters into vectors
            vint32m4_t v_multiplier = __riscv_vle32_v_i32m4(output_multiplier + current_out_c, vl);
            vint32m4_t v_shift = __riscv_vle32_v_i32m4(
                reinterpret_cast<const int32_t*>(output_shift) + current_out_c, vl);

            // Requantize the accumulated values using the fully vectorized helper.
            vint32m4_t v_res32 = RequantizeVectorPerChannelS32(
                v_acc_s32, v_multiplier, v_shift,
                output_offset, output_activation_min, output_activation_max, vl);
            
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
        while (current_out_c < static_cast<size_t>(output_depth))
        {
            // Set vector length for processing multiple output channels
            size_t vl = __riscv_vsetvl_e32m4(output_depth - current_out_c);

            // Initialize accumulator vector with biases
            vint32m4_t v_acc_s32 = bias_data
                ? __riscv_vle32_v_i32m4(bias_data + current_out_c, vl)
                : __riscv_vmv_v_x_i32m4(0, vl);

            // Loop over accumulation depth to compute 'vl' dot products in parallel
            for (int d = 0; d < accum_depth; ++d)
            {
                int16_t s_input_val_s16 = static_cast<int16_t>(input_batch_ptr[d]) + s_input_offset_s16;
                const int8_t* filter_col_ptr = filter_data + current_out_c * accum_depth + d;
                ptrdiff_t filter_stride = accum_depth * sizeof(int8_t);
                vint8m1_t v_filter_s8 = __riscv_vlse8_v_i8m1(filter_col_ptr, filter_stride, vl);
                vint16m2_t v_filter_s16 = __riscv_vsext_vf2_i16m2(v_filter_s8, vl);
                vint16m2_t v_filter_plus_offset_s16 = __riscv_vadd_vx_i16m2(v_filter_s16, s_filter_offset_s16, vl);
                v_acc_s32 = __riscv_vwmacc_vx_i32m4(v_acc_s32, s_input_val_s16, v_filter_plus_offset_s16, vl);
            }

            const int effective_right_shift = 31 - output_shift;
            vint32m4_t v_res32 = RequantizeVectorPerTensorS32(
                v_acc_s32,
                output_multiplier,
                effective_right_shift,
                output_offset,
                output_activation_min,
                output_activation_max,
                vl);

            // Narrow result to int8 and store
            vint16m2_t v_res16 = __riscv_vnclip_wx_i16m2(v_res32, 0, __RISCV_VXRM_RNU, vl);
            vint8m1_t v_out_s8 = __riscv_vnclip_wx_i8m1(v_res16, 0, __RISCV_VXRM_RNU, vl);
            __riscv_vse8_v_i8m1(output_batch_ptr + current_out_c, v_out_s8, vl);

            // Advance to the next block of output channels
            current_out_c += vl;
        }
    }
}