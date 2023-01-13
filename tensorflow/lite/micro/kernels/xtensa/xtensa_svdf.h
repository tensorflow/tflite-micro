/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SVDF_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SVDF_H_

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/svdf.h"

namespace tflite {
#if defined(HIFIMINI)
TfLiteStatus EvalIntegerSvdfHifimini(
    TfLiteContext* context, TfLiteNode* node,
    const TfLiteEvalTensor* input_tensor,
    const TfLiteEvalTensor* weights_feature_tensor,
    const TfLiteEvalTensor* weights_time_tensor,
    const TfLiteEvalTensor* bias_tensor, const TfLiteSVDFParams* params,
    TfLiteEvalTensor* activation_state_tensor, TfLiteEvalTensor* output_tensor,
    OpDataSvdf data);
#endif  // HIFIMINI

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_SVDF_H_
