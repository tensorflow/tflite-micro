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

#include "tensorflow/lite/micro/kernels/hard_swish.h"

#include <cstdint>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kHardSwishInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kHardSwishOutputTensor);
  TFLITE_DCHECK(node->user_data != nullptr);
  const HardSwishParams& params =
      *static_cast<const HardSwishParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      HardSwishEvalReference<float>(params, input, output);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      HardSwishEvalReference<int8_t>(params, input, output);
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Unsupported type %s", TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_HARD_SWISH() {
  return tflite::micro::RegisterOp(HardSwishInit, HardSwishPrepare,
                                   HardSwishEval);
}

}  // namespace micro
}  // namespace tflite
