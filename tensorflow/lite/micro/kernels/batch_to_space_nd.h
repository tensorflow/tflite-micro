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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_BATCH_TO_SPACE_ND_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_BATCH_TO_SPACE_ND_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kBatchToSpaceNDInputTensor = 0;
inline constexpr int kBatchToSpaceNDBlockShapeTensor = 1;
inline constexpr int kBatchToSpaceNDCropsTensor = 2;
inline constexpr int kBatchToSpaceNDOutputTensor = 0;

inline constexpr int kBatchToSpaceNDMinDim = 3;
inline constexpr int kBatchToSpaceNDMaxDim = 4;

TfLiteStatus BatchToSpaceNDPrepareReference(TfLiteContext* context,
                                            TfLiteNode* node);
TfLiteStatus BatchToSpaceNDEvalReference(TfLiteContext* context,
                                         TfLiteNode* node);

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_BATCH_TO_SPACE_ND_H_
