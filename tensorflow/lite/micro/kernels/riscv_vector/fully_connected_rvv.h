#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_FULLY_CONNECTED_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_FULLY_CONNECTED_RVV_H_

#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/c/common.h"

using namespace tflite;

void FullyConnectedPerChannelRVV(
    const FullyConnectedParams& params,
    const int32_t* output_multiplier,
    const int* output_shift,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape,
    const int32_t* bias_data,
    const RuntimeShape& output_shape,
    int8_t* output_data);

void FullyConnectedRVV(
    const FullyConnectedParams& params,
    const RuntimeShape& input_shape,
    const int8_t* input_data,
    const RuntimeShape& filter_shape,
    const int8_t* filter_data,
    const RuntimeShape& bias_shape, 
    const int32_t* bias_data,
    const RuntimeShape& output_shape, 
    int8_t* output_data);

#endif