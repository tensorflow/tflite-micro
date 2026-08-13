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

#include "tensorflow/lite/kernels/internal/reference/slice.h"

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kSizeTensor = 2;
constexpr int kOutputTensor = 0;

const int kMaxDim = 5;

template <typename T>
TfLiteStatus GetValidatedBeginAndSizeVectors(const TfLiteEvalTensor* input,
                                             const TfLiteEvalTensor* begin,
                                             const TfLiteEvalTensor* size,
                                             const TfLiteEvalTensor* output,
                                             int32_t* begins, int32_t* sizes) {
  const int dimensions = input->dims->size;
  int offset = kMaxDim - dimensions;
  int64_t sliced_elements = 1;
  bool sliced_elements_overflow = false;
  for (int idx = 0; idx < dimensions; ++idx) {
    const T begin_value = tflite::micro::GetTensorData<T>(begin)[idx];
    const T size_value = tflite::micro::GetTensorData<T>(size)[idx];
    const int input_size = input->dims->data[idx];

    if (begin_value < 0 || begin_value > input_size) {
      return kTfLiteError;
    }

    const T remaining_size = static_cast<T>(input_size) - begin_value;
    const T resolved_size = size_value == -1 ? remaining_size : size_value;
    if (resolved_size < 0 || resolved_size > remaining_size) {
      return kTfLiteError;
    }

    if (resolved_size == 0) {
      sliced_elements = 0;
      sliced_elements_overflow = false;
    } else if (!sliced_elements_overflow) {
      if (sliced_elements >
          std::numeric_limits<int64_t>::max() / resolved_size) {
        sliced_elements_overflow = true;
      } else {
        sliced_elements *= static_cast<int64_t>(resolved_size);
      }
    }
    begins[offset + idx] = static_cast<int32_t>(begin_value);
    sizes[offset + idx] = static_cast<int32_t>(size_value);
  }
  return !sliced_elements_overflow &&
                 sliced_elements ==
                     tflite::micro::GetTensorShape(output).FlatSize()
             ? kTfLiteOk
             : kTfLiteError;
}

TfLiteStatus SlicePrepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TFLITE_DCHECK(input != nullptr);
  TfLiteTensor* begin =
      micro_context->AllocateTempInputTensor(node, kBeginTensor);
  TFLITE_DCHECK(begin != nullptr);
  TfLiteTensor* size =
      micro_context->AllocateTempInputTensor(node, kSizeTensor);
  TFLITE_DCHECK(size != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TFLITE_DCHECK(output != nullptr);

  // Ensure validity of input tensor and its dimension.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, begin->type, size->type);
  TF_LITE_ENSURE(context,
                 begin->type == kTfLiteInt32 || begin->type == kTfLiteInt64);
  TF_LITE_ENSURE_EQ(context, NumDimensions(begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(size), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(begin), NumDimensions(input));
  TF_LITE_ENSURE_EQ(context, NumElements(size), NumDimensions(input));
  TF_LITE_ENSURE(context, NumDimensions(input) <= kMaxDim);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(begin);
  micro_context->DeallocateTempTfLiteTensor(size);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus SliceEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* begin =
      tflite::micro::GetEvalInput(context, node, kBeginTensor);
  const TfLiteEvalTensor* size =
      tflite::micro::GetEvalInput(context, node, kSizeTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  tflite::SliceParams op_params;
  op_params.begin_count = kMaxDim;
  op_params.size_count = kMaxDim;
  for (int i = 0; i < kMaxDim; ++i) {
    op_params.begin[i] = 0;
    op_params.size[i] = 1;
  }

  if (begin->type == kTfLiteInt32) {
    TF_LITE_ENSURE_STATUS(GetValidatedBeginAndSizeVectors<int32_t>(
        input, begin, size, output, op_params.begin, op_params.size));
  } else if (begin->type == kTfLiteInt64) {
    TF_LITE_ENSURE_STATUS(GetValidatedBeginAndSizeVectors<int64_t>(
        input, begin, size, output, op_params.begin, op_params.size));
  } else {
    MicroPrintf("Begin tensor type %s (%d) not supported.",
                TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Slice<float>(op_params,
                                  tflite::micro::GetTensorShape(input),
                                  tflite::micro::GetTensorData<float>(input),
                                  tflite::micro::GetTensorShape(output),
                                  tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt32:
      reference_ops::Slice<int32_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int32_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int32_t>(output));
      break;
    case kTfLiteInt8:
      reference_ops::Slice<int8_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      break;
    case kTfLiteInt16:
      reference_ops::Slice<int16_t>(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      break;
    case kTfLiteBool:
      reference_ops::Slice<bool>(op_params,
                                 tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<bool>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<bool>(output));
      break;
    default:
      MicroPrintf("Input tensor type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SLICE() {
  return tflite::micro::RegisterOp(nullptr, SlicePrepare, SliceEval);
}

}  // namespace tflite
