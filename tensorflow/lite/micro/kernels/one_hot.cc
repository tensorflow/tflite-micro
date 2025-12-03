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
#include <stdint.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace ops {
namespace micro {
namespace one_hot {

constexpr int kIndicesTensor = 0;
constexpr int kDepthTensor = 1;
constexpr int kOnValueTensor = 2;
constexpr int kOffValueTensor = 3;
constexpr int kOutputTensor = 0;

namespace {  // 로컬 유틸 함수들
inline int NumElements(const TfLiteEvalTensor* t) {
  int count = 1;
  // TfLiteEvalTensor의 dims는 TfLiteIntArray* 타입입니다.
  for (int i = 0; i < t->dims->size; ++i) {
    count *= t->dims->data[i];
  }
  return count;
}
}  // namespace

// TfLiteNode에서 입력 (indices, depth, on_value, off_value) 및 출력 텐서
// (output) 를 가져옴 params->axis 를 읽어 실제로 Depth 차원이 들어갈 위치
// (Axis) 계산 Prepare과 Eval 함수 내에서 잠시 생성되었다가 사라짐 → Stack
// memory 사용 효율적
struct OneHotContext {
  OneHotContext(TfLiteContext* context, TfLiteNode* node) {
    indices = tflite::micro::GetEvalInput(context, node, kIndicesTensor);
    depth = tflite::micro::GetEvalInput(context, node, kDepthTensor);
    on_value = tflite::micro::GetEvalInput(context, node, kOnValueTensor);
    off_value = tflite::micro::GetEvalInput(context, node, kOffValueTensor);
    output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

    const auto* params =
        reinterpret_cast<TfLiteOneHotParams*>(node->builtin_data);
    const int indices_dims = indices->dims->size;
    axis = (params->axis == -1) ? indices_dims : params->axis;
    output_dims = indices_dims + 1;
    dtype = on_value->type;
  }

  const TfLiteEvalTensor* indices;
  const TfLiteEvalTensor* depth;  // 새로 생기는 One-hot 차원 크기
  const TfLiteEvalTensor* on_value;
  const TfLiteEvalTensor* off_value;
  TfLiteEvalTensor* output;

  int axis;
  int output_dims;
  TfLiteType dtype;
};

// 실제 연산 수행 함수
//
template <typename T, typename TI>
void OneHotComputeImpl(const OneHotContext& op_context) {
  int prefix_dim_size = 1;
  for (int i = 0; i < op_context.axis; ++i) {
    prefix_dim_size *= op_context.indices->dims->data[i];
  }
  if (prefix_dim_size == 0) {
    return;
  }

  const RuntimeShape indices_shape =
      tflite::micro::GetTensorShape(op_context.indices);
  const int suffix_dim_size = indices_shape.FlatSize() / prefix_dim_size;

  const int depth = *op_context.depth->data.i32;

  const T on_value = *tflite::micro::GetTensorData<T>(op_context.on_value);
  const T off_value = *tflite::micro::GetTensorData<T>(op_context.off_value);

  T* output_data = tflite::micro::GetTensorData<T>(op_context.output);
  const TI* indices_data = tflite::micro::GetTensorData<TI>(op_context.indices);

  for (int i = 0; i < prefix_dim_size; ++i) {
    for (int j = 0; j < depth; ++j) {
      for (int k = 0; k < suffix_dim_size; ++k, ++output_data) {
        *output_data =
            static_cast<int>(indices_data[i * suffix_dim_size + k]) == j
                ? on_value
                : off_value;
      }
    }
  }
}

template <typename T>
void OneHotCompute(const OneHotContext& op_context) {
  if (op_context.indices->type == kTfLiteInt64) {
    OneHotComputeImpl<T, int64_t>(op_context);
  } else {
    OneHotComputeImpl<T, int32_t>(op_context);
  }
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const OneHotContext& op_context) {
  TF_LITE_ENSURE(context, *op_context.depth->data.i32 >= 0);

  // depth 데이터 읽기
  const int depth_val =
      *tflite::micro::GetTensorData<int32_t>(op_context.depth);
  TF_LITE_ENSURE(context, depth_val >= 0);

  // Output Tensor 검증
  TF_LITE_ENSURE(context, op_context.output != nullptr);

  TF_LITE_ENSURE(context, op_context.output->dims != nullptr);

  // todo
  // TFLM에서는 Output Tensor의 dims가 이미 할당되어 있다고 가정합니다.
  // 하지만 모델이 생성될 때 계산된 dims와 현재 depth값으로 계산한 dims가
  // 일치하는지 확인은 필요합니다.
  const int expected_dims_size = op_context.output_dims;
  TF_LITE_ENSURE_EQ(context, op_context.output->dims->size, expected_dims_size);

  for (int i = 0; i < expected_dims_size; ++i) {
    int expected_dim_i;
    if (i < op_context.axis) {
      expected_dim_i = op_context.indices->dims->data[i];
    } else if (i == op_context.axis) {
      expected_dim_i = depth_val;
    } else {
      expected_dim_i = op_context.indices->dims->data[i - 1];
    }

    // TFLM 컴파일러(Offline Memory Planner)가 할당해둔 크기와 실제 계산 크기가
    // 다르면 에러
    TF_LITE_ENSURE_EQ(context, op_context.output->dims->data[i],
                      expected_dim_i);
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  OneHotContext op_context{context, node};
  TF_LITE_ENSURE(context, op_context.output != nullptr);

  switch (op_context.dtype) {
    case kTfLiteFloat32:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteInt64:
    case kTfLiteInt8:
    case kTfLiteUInt8:
    case kTfLiteBool:
      op_context.output->type = op_context.dtype;
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Unknown output data type: %s",
                         TfLiteTypeGetName(op_context.dtype));
      return kTfLiteError;
  }

  TF_LITE_ENSURE(context, op_context.indices->type == kTfLiteInt32 ||
                              op_context.indices->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, op_context.axis >= 0 &&
                              op_context.axis < op_context.output_dims);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.depth), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.on_value), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.off_value), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.on_value->type, op_context.dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.off_value->type,
                          op_context.dtype);

  // depth 텐서가 상수가 아니더라도, 테스트에서는 output shape를
  // 미리 지정해 두었으므로 여기서는 그냥 검증만 수행
  return ResizeOutputTensor(context, op_context);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OneHotContext op_context{context, node};

  switch (op_context.output->type) {
    case kTfLiteFloat32:
      OneHotCompute<float>(op_context);
      break;
    case kTfLiteInt32:
      OneHotCompute<int32_t>(op_context);
      break;
    case kTfLiteInt64:
      OneHotCompute<int64_t>(op_context);
      break;
    case kTfLiteInt8:
      OneHotCompute<int8_t>(op_context);
      break;
    case kTfLiteUInt8:
      OneHotCompute<uint8_t>(op_context);
      break;
    case kTfLiteBool:
      OneHotCompute<bool>(op_context);
      break;
    default:
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace one_hot

// 헤더에 선언된 Register_ONE_HOT 구현
const TFLMRegistration* Register_ONE_HOT() {
  static TFLMRegistration r = {};

  r.prepare = one_hot::Prepare;
  r.invoke = one_hot::Eval;

  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite