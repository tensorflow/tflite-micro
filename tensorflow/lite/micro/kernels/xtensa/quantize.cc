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

#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
TfLiteStatus EvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteUInt8: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<uint8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }

        default:
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
      break;
    }

    case kTfLiteInt8: {
      switch (output->type) {
        case kTfLiteUInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<uint8_t>(output));
          break;
        }

        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const int8_t* input_data_ptr;
          int8_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
          output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_requantize_asym8s_asym8s(
                  output_data_ptr, input_data_ptr, op_data->input_zero_point,
                  zero_point, op_data->requantize_output_shift,
                  op_data->requantize_output_multiplier, size),
              0);
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int8_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              zero_point, tflite::micro::GetTensorData<int16_t>(output));
          break;
        }

        case kTfLiteInt32: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const int8_t* input_data_ptr;
          int32_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
          output_data_ptr = tflite::micro::GetTensorData<int32_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_requantize_asym8s_asym32s(
                  output_data_ptr, input_data_ptr, op_data->input_zero_point,
                  zero_point, op_data->requantize_output_shift,
                  op_data->requantize_output_multiplier, size),
              0);
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt16: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym8s(
                                tflite::micro::GetTensorData<int8_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym16s(
                                tflite::micro::GetTensorData<int16_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        case kTfLiteInt32: {
          int size = ElementCount(*input->dims);
          TF_LITE_ENSURE_EQ(context,
                            xa_nn_elm_requantize_asym16s_asym32s(
                                tflite::micro::GetTensorData<int32_t>(output),
                                tflite::micro::GetTensorData<int16_t>(input),
                                op_data->input_zero_point,
                                op_data->quantization_params.zero_point,
                                op_data->requantize_output_shift,
                                op_data->requantize_output_multiplier, size),
                            0);
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteInt32: {
      switch (output->type) {
        case kTfLiteInt8: {
          int size = ElementCount(*input->dims);
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int32_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              op_data->quantization_params.zero_point,
              tflite::micro::GetTensorData<int8_t>(output));
          break;
        }

        case kTfLiteInt16: {
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          reference_ops::Requantize(
              tflite::micro::GetTensorData<int32_t>(input), size,
              op_data->requantize_output_multiplier,
              op_data->requantize_output_shift, op_data->input_zero_point,
              zero_point, tflite::micro::GetTensorData<int16_t>(output));
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    case kTfLiteFloat32: {
      switch (output->type) {
        case kTfLiteInt8: {
#if HIFI_VFPU
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const float* input_data_ptr;
          int8_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<float>(input);
          output_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_quantize_f32_asym8s(
                  output_data_ptr, input_data_ptr,
                  static_cast<float>(op_data->quantization_params.scale),
                  zero_point, size),
              0);
#else   // #if HIFI_VFPU
          reference_ops::AffineQuantize(
              op_data->quantization_params,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<float>(input),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
#endif  // #if HIFI_VFPU
          break;
        }

        case kTfLiteInt16: {
#if HIFI_VFPU
          int size = ElementCount(*input->dims);
          int32_t zero_point = op_data->quantization_params.zero_point;
          const float* input_data_ptr;
          int16_t* output_data_ptr;
          input_data_ptr = tflite::micro::GetTensorData<float>(input);
          output_data_ptr = tflite::micro::GetTensorData<int16_t>(output);

          TF_LITE_ENSURE_EQ(
              context,
              xa_nn_elm_quantize_f32_asym16s(
                  output_data_ptr, input_data_ptr,
                  static_cast<float>(op_data->quantization_params.scale),
                  zero_point, size),
              0);
#else   // #if HIFI_VFPU
          reference_ops::AffineQuantize(
              op_data->quantization_params,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<float>(input),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int16_t>(output));
#endif  // #if HIFI_VFPU
          break;
        }

        default: {
          MicroPrintf("Input %s, output %s not supported.",
                      TfLiteTypeGetName(input->type),
                      TfLiteTypeGetName(output->type));
          return kTfLiteError;
        }
      }
      break;
    }

    default: {
      MicroPrintf("Input %s, output %s not supported.",
                  TfLiteTypeGetName(input->type),
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);

  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);
  op_data->quantization_params.zero_point = output->params.zero_point;
  op_data->quantization_params.scale =
      static_cast<double>(output->params.scale);

  op_data->input_zero_point = input->params.zero_point;

  double effective_scale = static_cast<double>(input->params.scale) /
                           static_cast<double>(output->params.scale);
  QuantizeMultiplier(effective_scale, &op_data->requantize_output_multiplier,
                     &op_data->requantize_output_shift);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
  return EvalXtensa(context, node);
#else
  return EvalQuantizeReference(context, node);
#endif  // defined(HIFI3) || defined(HIFI4) || defined(HIFI5)
}

}  // namespace

TFLMRegistration Register_QUANTIZE() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
