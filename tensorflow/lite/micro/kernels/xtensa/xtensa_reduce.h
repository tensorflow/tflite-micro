/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_REDUCE_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_REDUCE_H_

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
struct XtensaReduceOpData {
  OpDataReduce reference_op_data;

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  int scratch_tensor_index;
  int32_t updated_multiplier;
  int32_t updated_shift;
  int resolved_axis[4];
  int num_resolved_axis;
  bool is_hifi_supported;
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

#if defined(VISION_P6)
  uint8_t* p_context;  // persistent lib context for this instance saved here
  uint32_t context_size;
#endif  // VISION_P6
};

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

TfLiteStatus ReducePrepareMeanHifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus ReducePrepareMaxHifi(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus ReduceEvalMeanHifi(TfLiteContext* context, TfLiteNode* node,
                                const XtensaReduceOpData& data);

TfLiteStatus ReduceEvalMaxHifi(TfLiteContext* context, TfLiteNode* node,
                               const XtensaReduceOpData& data);

#elif defined(VISION_P6)

TfLiteStatus ReducePrepareVision(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus ReduceEvalVision(const XtensaReduceOpData& data,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output);

#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

}  // namespace micro

using micro::XtensaReduceOpData;
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_XTENSA_XTENSA_REDUCE_H_
