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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

const int kMaxNumberOfAxis = 5;
const int kMaxNumberOfReducedAxis = 2;

namespace {

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node,
                           int32_t* multiplier, int* shift) {
  MicroContext* micro_context = GetMicroContext(context);

  // Inputs Tensor (dtype depends on quantization):
  // [0] = Input
  // [1] = Axis
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);

  // Outputs Tensor (dtype depends on quantization):
  // [0] = Output

  // Validate number of inputs and outputs
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Validate axis type
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);
  TF_LITE_ENSURE(context, axis != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, axis->type, kTfLiteInt32);

  if (input->type == kTfLiteInt8) {
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, multiplier, shift);
    micro_context->DeallocateTempTfLiteTensor(output);
  }
  micro_context->DeallocateTempTfLiteTensor(axis);
  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}

void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16_t>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
  op_params->axis_count = axis_count;
}

template <typename T>
TfLiteStatus QuantizedMeanOrSum(TfLiteContext* context, TfLiteNode* node,
                                int* temp_index, int* resolved_axis,
                                int32_t* temp_sum, OpDataReduce* op_data,
                                bool compute_sum) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  bool result = reference_ops::QuantizedMeanOrSumExtraArgs<T, int32_t>(
      tflite::micro::GetTensorData<T>(input), op_data->input_zp,
      op_data->input_scale, &input->dims->data[0], input->dims->size,
      tflite::micro::GetTensorData<T>(output), op_data->output_scale,
      op_data->multiplier, op_data->shift, op_data->output_zp,
      &output->dims->data[0], output->dims->size,
      tflite::micro::GetTensorData<int>(axis), op_data->num_axis,
      params->keep_dims, temp_index, resolved_axis, temp_sum, compute_sum);
  TF_LITE_ENSURE(context, result);

  return kTfLiteOk;
}

template <typename integer_type>
TfLiteStatus EvalIntegerMean(TfLiteContext* context, TfLiteNode* node,
                             int num_axis, OpDataReduce* op_data,
                             int* temp_index, int* resolved_axis) {
  int32_t* temp_sum = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));

  QuantizedMeanOrSum<integer_type>(context, node, temp_index, resolved_axis,
                                   temp_sum, op_data, /*compute_sum=*/false);

  return kTfLiteOk;
}

enum MinMaxEvalType { kEvalMin, kEvalMax };

template <typename T>
struct MinMaxReducerCompare {
  MinMaxReducerCompare() = delete;
  MinMaxReducerCompare(MinMaxEvalType evalType) : type_(evalType){};

  constexpr T initialValue() const {
    return (type_ == kEvalMin) ? std::numeric_limits<T>::max()
                               : std::numeric_limits<T>::lowest();
  }

  // should be able to use "auto" keyword here, but GCC and Clang blow a fuse
  T (*compare())(const T, const T) {
    if (type_ == kEvalMin) {
      return [](const T current, const T in) -> T {
        return (in < current) ? in : current;
      };
    } else {
      return [](const T current, const T in) -> T {
        return (in > current) ? in : current;
      };
    }
  }

 private:
  MinMaxEvalType type_;
};

TfLiteStatus EvalMinMaxHelper(TfLiteContext* context, TfLiteNode* node,
                              OpDataReduce* op_data, MinMaxEvalType evalType) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  // Interpret an axis tensor with null dimensions as a scalar
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int* temp_buffer = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->temp_buffer_idx));
  int* resolved_axis = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->resolved_axis_idx));
  switch (input->type) {
    case kTfLiteFloat32: {
      MinMaxReducerCompare<float> reducer(evalType);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<float>(
              tflite::micro::GetTensorData<float>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<float>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              reducer.initialValue(), reducer.compare()));
    } break;
    case kTfLiteInt8: {
      MinMaxReducerCompare<int8_t> reducer(evalType);
      TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                        static_cast<double>(op_data->output_scale));
      TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<int8_t>(
              tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              reducer.initialValue(), reducer.compare()));
    } break;
    default:
      MicroPrintf("Only float32 and int8 types are supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteStatus PrepareMinMaxHelper(TfLiteContext* context, TfLiteNode* node,
                                 OpDataReduce* op_data) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node, &op_data->multiplier,
                                           &op_data->shift));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);

  op_data->input_zp = input->params.zero_point;
  op_data->input_scale = input->params.scale;
  op_data->output_zp = output->params.zero_point;
  op_data->output_scale = output->params.scale;
  op_data->num_output_elements = NumElements(output);

  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->temp_buffer_idx);
  context->RequestScratchBufferInArena(
      context, sizeof(int) * static_cast<int>(ElementCount(*axis->dims)),
      &op_data->resolved_axis_idx);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus PrepareMeanOrSumHelper(TfLiteContext* context, TfLiteNode* node,
                                    OpDataReduce* op_data) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &op_data->shift);
  }

  int output_size = NumElements(output);
  op_data->num_axis = NumElements(axis);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    context->RequestScratchBufferInArena(context, output_size * sizeof(int32_t),
                                         &op_data->temp_buffer_idx);
    op_data->input_zp = input->params.zero_point;
    op_data->input_scale = input->params.scale;
    op_data->output_zp = output->params.zero_point;
    op_data->output_scale = output->params.scale;
  }

  TF_LITE_ENSURE_OK(
      context,
      PrepareSimple(context, node, &(op_data->multiplier), &(op_data->shift)));
  // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus EvalMeanHelper(TfLiteContext* context, TfLiteNode* node,
                            OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);

  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::MeanParams op_params;
      ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis,
                  &op_params);

      // Special case mean implementation exists for 4D mean across axes 1
      // and 2.
      bool special_case_4d_axes_1_and_2 =
          input->dims->size == 4 && op_params.axis_count == 2 &&
          ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
           (op_params.axis[0] == 2 && op_params.axis[1] == 1));

      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (params->keep_dims && special_case_4d_axes_1_and_2) {
        reference_ops::Mean(op_params, tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
      } else {
        TF_LITE_ENSURE(
            context,
            reference_ops::Mean(
                tflite::micro::GetTensorData<float>(input), input->dims->data,
                input->dims->size, tflite::micro::GetTensorData<float>(output),
                output->dims->data, output->dims->size,
                tflite::micro::GetTensorData<int>(axis), num_axis,
                params->keep_dims, temp_index, resolved_axis,
                tflite::micro::GetTensorData<float>(output)));
      }
    } break;
    case kTfLiteInt8: {
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int8_t>(context, node, num_axis, op_data,
                                           temp_index, resolved_axis));
    } break;
    case kTfLiteInt16: {
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int16_t>(context, node, num_axis, op_data,
                                            temp_index, resolved_axis));
    } break;
    default:
      TF_LITE_ENSURE_MSG(context, false,
                         "Currently, only float32, int8 or int16 input type "
                         "is supported.");
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMaxHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  return EvalMinMaxHelper(context, node, op_data, kEvalMax);
}

TfLiteStatus EvalMinHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  return EvalMinMaxHelper(context, node, op_data, kEvalMin);
}

TfLiteStatus EvalSumHelper(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  // Interpret an axis tensor with null dimensions as a scalar.
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxis];
  int resolved_axis[kMaxNumberOfReducedAxis];

  switch (input->type) {
    case kTfLiteFloat32: {
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<float>(
              tflite::micro::GetTensorData<float>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<float>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_index, resolved_axis, /*init_value=*/0.f,
              [](const float current, const float in) -> float {
                return in + current;
              }));
    } break;
    case kTfLiteInt8: {
      int32_t* temp_sum = static_cast<int32_t*>(
          context->GetScratchBuffer(context, op_data->temp_buffer_idx));
      QuantizedMeanOrSum<int8_t>(context, node, temp_index, resolved_axis,
                                 temp_sum, op_data, /*compute_sum=*/true);
    } break;
    case kTfLiteInt16: {
      int32_t* temp_sum = static_cast<int32_t*>(
          context->GetScratchBuffer(context, op_data->temp_buffer_idx));
      QuantizedMeanOrSum<int16_t>(context, node, temp_index, resolved_axis,
                                  temp_sum, op_data, /*compute_sum=*/true);
    } break;
    default:
      MicroPrintf("Only float32, int8, and int16 types are supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace tflite
