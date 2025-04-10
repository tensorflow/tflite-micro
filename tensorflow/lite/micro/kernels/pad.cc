/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include <string.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

TfLiteStatus PadEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);
  const TfLiteEvalTensor* constant_values =
      NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, /*index=*/2)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  switch (input->type) {
    case kTfLiteFloat32: {
      float pad_value =
          constant_values == nullptr
              ? 0.f
              : *tflite::micro::GetTensorData<float>(constant_values);
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<float>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<float>(output));
      }
    } break;
    case kTfLiteInt8: {
      int8_t pad_value;
      if (constant_values == nullptr) {
        pad_value = static_cast<uint8_t>(data->output_zero_point);
      } else {
        pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
      }
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int8_t>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int8_t>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<int8_t>(output));
      }
    } break;
    case kTfLiteInt16: {
      int16_t pad_value =
          constant_values == nullptr
              ? 0
              : *tflite::micro::GetTensorData<int16_t>(constant_values);
      reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int16_t>(input),
                         &pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int16_t>(output));
    } break;
    case kTfLiteInt32: {
      int32_t pad_value =
          constant_values == nullptr
              ? 0
              : *tflite::micro::GetTensorData<int32_t>(constant_values);
      reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int32_t>(input),
                         &pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int32_t>(output));
    } break;
    default:

      MicroPrintf("Type %s not currently supported by Pad.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_PAD() {
  return tflite::micro::RegisterOp(PadInit, PadPrepare, PadEval);
}

// Also register Pad as PadV2.
TFLMRegistration Register_PADV2() {
  return tflite::micro::RegisterOp(PadInit, PadPrepare, PadEval);
}

}  // namespace tflite
