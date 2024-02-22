/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/sub.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

void* SubInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataSub));
}

void EvalSub(TfLiteContext* context, TfLiteNode* node, TfLiteSubParams* params,
             const OpDataSub* data, const TfLiteEvalTensor* input1,
             const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::ArithmeticParams op_params;
  SetActivationParams(output_activation_min, output_activation_max, &op_params);
  if (data->requires_broadcast) {
    tflite::reference_ops::BroadcastSubSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  } else {
    tflite::reference_ops::SubWithActivation(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  }
}

TfLiteStatus EvalSubQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteSubParams* params, const OpDataSub* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);
  // TODO(b/259724572): vision_p6 and hifi code path is getting very confusing.
  // Let's separate them into two different files.
#if !(defined(HIFI3) || defined(HIFI4) || defined(HIFI5))
  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);
#endif  // !(defined(HIFI3) || defined(HIFI4))

  switch (output->type) {
    case kTfLiteInt8: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      int err;
      const RuntimeShape extended_input1_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input1));
      const RuntimeShape extended_input2_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input2));
      const RuntimeShape extended_output_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(output));
      const int* input1_dims = extended_input1_shape.DimsData();
      const int* input2_dims = extended_input2_shape.DimsData();
      const int* output_dims = extended_output_shape.DimsData();
      // TODO(b/259724572): Refactor the following block of code.
      int b;
      int inp1_off = 0;
      int out_off;
      out_off =
          output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];
      if (input1_dims[0] > 1) {
        inp1_off =
            input1_dims[1] * input1_dims[2] * input1_dims[3] * input1_dims[4];
      }

      for (b = 0; b < output_dims[0]; b++) {
        err = xa_nn_elm_sub_broadcast_4D_asym8sxasym8s_asym8s(
            tflite::micro::GetTensorData<int8_t>(output) + b * out_off,
            output_dims + 1, op_params.output_offset, op_params.output_shift,
            op_params.output_multiplier, op_params.quantized_activation_min,
            op_params.quantized_activation_max,
            tflite::micro::GetTensorData<int8_t>(input1) + b * inp1_off,
            input1_dims + 1, op_params.input1_offset, op_params.input1_shift,
            op_params.input1_multiplier,
            tflite::micro::GetTensorData<int8_t>(input2), input2_dims + 1,
            op_params.input2_offset, op_params.input2_shift,
            op_params.input2_multiplier, op_params.left_shift);

        TF_LITE_ENSURE(context, err == 0);
      }
#else   // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      if (need_broadcast) {
        tflite::reference_ops::BroadcastQuantSubSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
        tflite::reference_ops::Sub(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      }
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      break;
    }
    case kTfLiteInt16: {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      int err;
      const RuntimeShape extended_input1_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input1));
      const RuntimeShape extended_input2_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(input2));
      const RuntimeShape extended_output_shape =
          RuntimeShape::ExtendedShape(5, tflite::micro::GetTensorShape(output));
      const int* input1_dims = extended_input1_shape.DimsData();
      const int* input2_dims = extended_input2_shape.DimsData();
      const int* output_dims = extended_output_shape.DimsData();
      int b;
      int inp1_off = 0;
      int out_off;
      out_off =
          output_dims[1] * output_dims[2] * output_dims[3] * output_dims[4];
      if (input1_dims[0] > 1) {
        inp1_off =
            input1_dims[1] * input1_dims[2] * input1_dims[3] * input1_dims[4];
      }

      for (b = 0; b < output_dims[0]; b++) {
        err = xa_nn_elm_sub_broadcast_4D_asym16sxasym16s_asym16s(
            tflite::micro::GetTensorData<int16_t>(output) + b * out_off,
            output_dims + 1, op_params.output_offset, op_params.output_shift,
            op_params.output_multiplier, op_params.quantized_activation_min,
            op_params.quantized_activation_max,
            tflite::micro::GetTensorData<int16_t>(input1) + b * inp1_off,
            input1_dims + 1, op_params.input1_offset, op_params.input1_shift,
            op_params.input1_multiplier,
            tflite::micro::GetTensorData<int16_t>(input2), input2_dims + 1,
            op_params.input2_offset, op_params.input2_shift,
            op_params.input2_multiplier, op_params.left_shift);

        TF_LITE_ENSURE(context, err == 0);
      }
#else   // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      if (need_broadcast) {
        tflite::reference_ops::BroadcastQuantSubSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int16_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int16_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        tflite::reference_ops::Sub(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int16_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int16_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      }
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
      break;
    }
    default:
      MicroPrintf("Quantized type %s not currently supported.",
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SubEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSubParams*>(node->builtin_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kSubInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kSubInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSubOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataSub& data = *(static_cast<const OpDataSub*>(node->user_data));

  if (output->type == kTfLiteFloat32) {
    EvalSub(context, node, params, &data, input1, input2, output);
  } else if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(context, EvalSubQuantized(context, node, params, &data,
                                                input1, input2, output));
  } else {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(output->type),
                output->type);
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TFLMRegistration Register_SUB() {
  return tflite::micro::RegisterOp(SubInit, SubPrepare, SubEval);
}

}  // namespace tflite
