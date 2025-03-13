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

#include <limits>

#include "Include/arm_nn_types.h"
#include "Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pad.h"

namespace tflite {
namespace {

TfLiteStatus PadEvalInt8(TfLiteContext* context, TfLiteNode* node) {
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

  int8_t pad_value;
  if (constant_values == nullptr) {
    pad_value = static_cast<uint8_t>(data->output_zero_point);
  } else {
    pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
  }
  const int8_t* input_ptr = tflite::micro::GetTensorData<int8_t>(input);
  int8_t* output_ptr = tflite::micro::GetTensorData<int8_t>(output);

  const RuntimeShape d = tflite::micro::GetTensorShape(input);
  const cmsis_nn_dims input_size = {d.Dims(0), d.Dims(1), d.Dims(2), d.Dims(3)};

  const PadParams p = data->params;
  const cmsis_nn_dims pre_pad = {p.left_padding[0], p.left_padding[1],
                                 p.left_padding[2], p.left_padding[3]};
  const cmsis_nn_dims post_pad = {p.right_padding[0], p.right_padding[1],
                                  p.right_padding[2], p.right_padding[3]};

  arm_pad_s8(input_ptr, output_ptr, pad_value, &input_size, &pre_pad,
             &post_pad);

  return kTfLiteOk;
}

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
      PadEvalInt8(context, node);
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

TFLMRegistration Register_PAD_INT8() {
  return tflite::micro::RegisterOp(PadInit, PadPrepare, PadEvalInt8);
}

}  // namespace tflite
