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

#include "tensorflow/lite/micro/kernels/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
#if defined(HIFI5)
  auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
  const OpDataPooling* reference_op_data = &(op_data->reference_op_data);
#else
  const OpDataPooling* reference_op_data =
      static_cast<const OpDataPooling*>(node->user_data);
#endif

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32: {
      AveragePoolingEvalFloat(context, node, params, reference_op_data, input,
                              output);
      break;
    }
    case kTfLiteInt8: {
#if defined(HIFI5)
      AverageEvalQuantizedHifi(context, node, params, op_data, input, output);
#elif defined(VISION_P6)
      const auto& op_data =
          *(reinterpret_cast<XtensaOpDataPooling*>(node->user_data));
      PoolEvalVision(context, node, *params, op_data, input, output);
#else
      AveragePoolingEvalQuantized<int8_t>(context, node, params,
                                          reference_op_data, input, output);
#endif
      break;
    }
    case kTfLiteInt16: {
      AveragePoolingEvalQuantized<int16_t>(context, node, params,
                                           reference_op_data, input, output);
      break;
    }
    default: {
      MicroPrintf("Input type %s is not currently supported",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
#if defined(HIFI5)
  auto* op_data = static_cast<const XtensaOpDataPooling*>(node->user_data);
  const OpDataPooling* reference_op_data = &(op_data->reference_op_data);
#else
  const OpDataPooling* reference_op_data =
      static_cast<const OpDataPooling*>(node->user_data);
#endif

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      MaxPoolingEvalFloat(context, node, params, reference_op_data, input,
                          output);
      break;
    }
    case kTfLiteInt8: {
#if defined(HIFI5)
      MaxEvalQuantizedHifi(context, node, params, op_data, input, output);
#elif defined(VISION_P6)
      const auto& op_data =
          *(reinterpret_cast<XtensaOpDataPooling*>(node->user_data));
      PoolEvalVision(context, node, *params, op_data, input, output);
#else
      MaxPoolingEvalQuantized<int8_t>(context, node, params, reference_op_data,
                                      input, output);
#endif
      break;
    }
    case kTfLiteInt16: {
      MaxPoolingEvalQuantized<int16_t>(context, node, params, reference_op_data,
                                       input, output);
      break;
    }
    default: {
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D() {
#if defined(HIFI5)
  return tflite::micro::RegisterOp(XtensaPoolingInit, AveragePrepareHifi,
                                   AverageEval);
#elif defined(VISION_P6)
  return tflite::micro::RegisterOp(XtensaPoolingInit, AvgPoolingPrepareVision,
                                   AverageEval);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare,
                                   AverageEval);
#endif
}

TFLMRegistration Register_MAX_POOL_2D() {
#if defined(HIFI5)
  return tflite::micro::RegisterOp(XtensaPoolingInit, MaxPrepareHifi, MaxEval);
#elif defined(VISION_P6)
  return tflite::micro::RegisterOp(XtensaPoolingInit, MaxPoolingPrepareVision,
                                   MaxEval);
#else
  return tflite::micro::RegisterOp(XtensaPoolingInit, PoolingPrepare, MaxEval);
#endif
}

TFLMRegistration Register_AVERAGE_POOL_2D_INT16() {
  return Register_AVERAGE_POOL_2D();
}

TFLMRegistration Register_MAX_POOL_2D_INT16() { return Register_MAX_POOL_2D(); }

}  // namespace tflite
