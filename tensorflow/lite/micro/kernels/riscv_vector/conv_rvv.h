// tensorflow/lite/micro/kernels/riscv_vector/conv_rvv.h
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_CONV_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_CONV_RVV_H_

#include <cstdint>
#include <cstddef>

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
    const uint16_t pad_width);

#endif