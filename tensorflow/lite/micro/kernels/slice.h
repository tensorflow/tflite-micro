/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_SLICE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_SLICE_H_

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kSliceInputTensor = 0;
inline constexpr int kSliceBeginTensor = 1;
inline constexpr int kSliceSizeTensor = 2;
inline constexpr int kSliceOutputTensor = 0;

inline constexpr int kSliceMaxDim = 5;

template <typename T>
void GetSliceBeginAndSizeVectors(int dimensions, const TfLiteEvalTensor* begin,
                                 const TfLiteEvalTensor* size, int32_t* begins,
                                 int32_t* sizes) {
  int offset = kSliceMaxDim - dimensions;
  for (int idx = 0; idx < dimensions; ++idx) {
    begins[offset + idx] = tflite::micro::GetTensorData<T>(begin)[idx];
    sizes[offset + idx] = tflite::micro::GetTensorData<T>(size)[idx];
  }
}

TfLiteStatus PopulateSliceParams(const TfLiteEvalTensor* input,
                                 const TfLiteEvalTensor* begin,
                                 const TfLiteEvalTensor* size,
                                 SliceParams* op_params);
TfLiteStatus SlicePrepareReference(TfLiteContext* context, TfLiteNode* node);
TfLiteStatus SliceEvalReference(TfLiteContext* context, TfLiteNode* node);

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_SLICE_H_
