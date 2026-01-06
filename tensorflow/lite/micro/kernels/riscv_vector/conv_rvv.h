#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_CONV_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_CONV_RVV_H_

#include <cstdint>
#include <cstddef>

#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

using namespace tflite;

void ConvPerChannelRVV(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data);

void DepthwiseConvPerChannelRVV(const DepthwiseParams& params,
    const int32_t* output_multiplier,
    const int32_t* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data);

#endif // TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_CONV_RVV_H_