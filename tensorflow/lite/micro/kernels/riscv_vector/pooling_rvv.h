#ifndef TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_POOLING_RVV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_RISCV_VECTOR_POOLING_RVV_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"

using namespace tflite;

void MaxPool8BitRVV(const PoolParams& params, const RuntimeShape& input_shape,
                    const int8_t* input_data, const RuntimeShape& output_shape,
                    int8_t* output_data);

#endif