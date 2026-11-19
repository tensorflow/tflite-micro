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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_DEPTH_TO_SPACE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_DEPTH_TO_SPACE_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {

inline constexpr int kDepthToSpaceInputTensor = 0;
inline constexpr int kDepthToSpaceOutputTensor = 0;

inline constexpr int kDepthToSpaceBatchRank = 0;
inline constexpr int kDepthToSpaceHeightRank = 1;
inline constexpr int kDepthToSpaceWidthRank = 2;
inline constexpr int kDepthToSpaceDepthRank = 3;

TfLiteStatus DepthToSpacePrepareReference(TfLiteContext* context,
                                          TfLiteNode* node);
TfLiteStatus DepthToSpaceEvalReference(TfLiteContext* context,
                                       TfLiteNode* node);

}  // namespace micro
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_DEPTH_TO_SPACE_H_
