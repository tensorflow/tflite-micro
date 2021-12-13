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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

struct OpDataMirrorPad {
  int input_dims;
  int output_size;
  int offset;
  int output_dims_num_elements_buffer_index;
  int input_dims_num_elements_buffer_index;
};

// Nil value for paddingMode/offset.
const int kUnsetOffset = -1;

// Wrapper for params passed to the Eval<T> function.
template <typename T>
struct EvalData {
  const TfLiteEvalTensor* padding_matrix = nullptr;
  const TfLiteIntArray* input_dims = nullptr;
  // Holds number of elements at the nth dimension.
  // value at last dimension = 1, at second to last = sizeof last dimension.
  int* output_dims_num_elements;
  int* input_dims_num_elements;

  const T* input_data = nullptr;

  int offset = kUnsetOffset;
  T* output_data = nullptr;
  int num_dims = 0;
};

// Helper method that fills the left and right pads.
template <typename T>
inline void GetPadding(const T* data, int offset, int64_t* left_pad,
                       int64_t* right_pad) {
  *left_pad = static_cast<int64_t>(*(data + offset * 2));
  *right_pad = static_cast<int64_t>(*(data + offset * 2 + 1));
}

inline void GetPadding(const TfLiteEvalTensor* padding_matrix, int dimension,
                       int64_t* left_pad, int64_t* right_pad) {
  switch (padding_matrix->type) {
    case kTfLiteInt32:
      GetPadding(padding_matrix->data.i32, dimension, left_pad, right_pad);
      break;
    case kTfLiteInt64:
      GetPadding(padding_matrix->data.i64, dimension, left_pad, right_pad);
      break;
    default:
      return;
  }
}

// Given dimension index and the left/right padding.
// Returns the corresponding dimension in the input array.
inline int GetInputDimension(int padded_dimension, int left_pad, int right_pad,
                             int input_dim_size, int offset) {
  if (padded_dimension < left_pad) {
    const int original_ind = left_pad + offset - 1;
    return original_ind - (std::min(padded_dimension, original_ind - offset));
  }
  padded_dimension -= left_pad;
  if (padded_dimension >= input_dim_size) {
    padded_dimension -= input_dim_size;
    const int original_ind = input_dim_size - (1 + offset);
    return original_ind - std::min(padded_dimension, original_ind);
  }
  return padded_dimension;
}

// Given and index in output array, returns the index of the value
// in input array.
template <typename T>
int GetFlatIndex(int index, EvalData<T>* eval_data) {
  int flat_index = 0;
  int64_t left_pad = 0, right_pad = 0, dimension_index, index_in_input;

  for (int i = 0; i < eval_data->num_dims; ++i) {
    switch (eval_data->padding_matrix->type) {
      case kTfLiteInt32:
        GetPadding(eval_data->padding_matrix->data.i32, i, &left_pad,
                   &right_pad);
        // printf("correct  %d %d\n", left_pad, right_pad);
        break;
      case kTfLiteInt64:
        GetPadding(eval_data->padding_matrix->data.i64, i, &left_pad,
                   &right_pad);
        break;
      default:
        break;
    }
    dimension_index = index / (eval_data->output_dims_num_elements)[i];

    index_in_input =
        GetInputDimension(dimension_index, left_pad, right_pad,
                          eval_data->input_dims->data[i], eval_data->offset);

    flat_index += index_in_input * (eval_data->input_dims_num_elements)[i];
    index %= (eval_data->output_dims_num_elements)[i];
  }

  return flat_index;
}

template <typename T>
struct MirrorPadWorkerTask {
  MirrorPadWorkerTask(EvalData<T>* eval_data_, int start_, int end_)
      : eval_data(eval_data_), start(start_), end(end_) {}
  void Run() {
    auto* input_data = eval_data->input_data;
    auto* output_data = eval_data->output_data;
    for (int i = start; i < end; ++i) {
      output_data[i] = input_data[GetFlatIndex(i, eval_data)];
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TfLiteStatus status = kTfLiteOk;
  const OpDataMirrorPad* data =
      static_cast<const OpDataMirrorPad*>(node->user_data);

  const TfLiteEvalTensor* input_tensor =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* padding_matrix =
      tflite::micro::GetEvalInput(context, node, 1);

  TfLiteEvalTensor* output_tensor =
      tflite::micro::GetEvalOutput(context, node, 0);
  const int input_dims = data->input_dims;
  const int output_size = data->output_size;

  int* input_dims_num_elements = (int*)context->GetScratchBuffer(
      context, data->input_dims_num_elements_buffer_index);
  int* output_dims_num_elements = (int*)context->GetScratchBuffer(
      context, data->output_dims_num_elements_buffer_index);

  for (int i = 0; i < input_dims; i++) {
    output_dims_num_elements[i] = 1;
    input_dims_num_elements[i] = 1;
  }

  for (int i = input_dims - 2; i >= 0; i--) {
    output_dims_num_elements[i] =
        output_dims_num_elements[i + 1] * output_tensor->dims->data[i + 1];

    input_dims_num_elements[i] =
        input_dims_num_elements[i + 1] * input_tensor->dims->data[i + 1];
  }

#define TF_LITE_MIRROR_PAD(type)                                             \
  EvalData<type> eval_data;                                                  \
  eval_data.input_data = tflite::micro::GetTensorData<type>(input_tensor);   \
  eval_data.input_dims = input_tensor->dims;                                 \
  eval_data.output_dims_num_elements = output_dims_num_elements;             \
  eval_data.input_dims_num_elements = input_dims_num_elements;               \
  eval_data.num_dims = input_dims;                                           \
  eval_data.offset = data->offset;                                           \
  eval_data.output_data = tflite::micro::GetTensorData<type>(output_tensor); \
  eval_data.padding_matrix = padding_matrix;                                 \
  MirrorPadWorkerTask<type> task(&eval_data, 0, output_size);                \
  task.Run();

  switch (output_tensor->type) {
    case kTfLiteFloat32: {
      TF_LITE_MIRROR_PAD(float);
      break;
    }
    case kTfLiteInt8: {
      TF_LITE_MIRROR_PAD(int8_t);
      break;
    }
    default:
      status = kTfLiteError;
      break;
  }

#undef TF_LITE_MIRROR_PAD

  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpDataMirrorPad));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpDataMirrorPad* data = static_cast<OpDataMirrorPad*>(node->user_data);

  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  const TfLiteTensor* padding_matrix = GetInput(context, node, 1);
  TfLiteTensor* output_tensor = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(padding_matrix), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 0),
                    NumDimensions(input_tensor));
  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);
  if (params == nullptr) {
    return kTfLiteError;
  }

  data->offset =
      params->mode != TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect ? 0
                                                                           : 1;
  data->input_dims = NumDimensions(input_tensor);
  data->output_size = NumElements(output_tensor);

  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, data->input_dims * sizeof(int),
      &data->output_dims_num_elements_buffer_index));
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, data->input_dims * sizeof(int),
      &data->input_dims_num_elements_buffer_index));

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_MIRROR_PAD() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
