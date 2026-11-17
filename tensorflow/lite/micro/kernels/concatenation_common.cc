/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/kernels/concatenation.h"
#include "tensorflow/lite/micro/kernels/internal/reference/concatenation.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

inline int CalculatePositiveAxis(int axis, const TfLiteTensor* output_tensor) {
  if (axis >= 0) {
    return axis;
  }
  return NumDimensions(output_tensor) + axis;
}

inline void GetAllInputTensorShapes(
    const TfLiteContext* context, const TfLiteNode* node,
    RuntimeShape all_shapes[kConcatenationMaxInputNum]) {
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    RuntimeShape shape = tflite::micro::GetTensorShape(t);
    all_shapes[i].ReplaceWith(shape.DimensionsCount(), shape.DimsData());
  }
}

inline void GetShapesPointers(const RuntimeShape* shapes, size_t num,
                              const RuntimeShape* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &shapes[i];
  }
}

template <typename T>
inline void GetAllInputTensorData(
    const TfLiteContext* context, const TfLiteNode* node,
    const T* all_data[kConcatenationMaxInputNum]) {
#ifdef USE_TFLM_COMPRESSION
  const ConcatenationOpData* data =
      static_cast<const ConcatenationOpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);
#endif  // USE_TFLM_COMPRESSION

  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
#ifdef USE_TFLM_COMPRESSION
    const CompressionTensorData* comp_td =
        micro_context->GetTensorCompressionData(node, i);
    all_data[i] = tflite::micro::GetTensorData<T>(micro_context, t, comp_td,
                                                  data->scratch_indices[i]);
#else   // USE_TFLM_COMPRESSION
    all_data[i] = tflite::micro::GetTensorData<T>(t);
#endif  // USE_TFLM_COMPRESSION
  }
}

template <typename DataType>
void EvalUnquantized(TfLiteContext* context, TfLiteNode* node) {
  RuntimeShape inputs_shape[kConcatenationMaxInputNum];
  const RuntimeShape* inputs_shape_ptr[kConcatenationMaxInputNum];
  const DataType* inputs_data[kConcatenationMaxInputNum];
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);
  const ConcatenationOpData* data =
      static_cast<const ConcatenationOpData*>(node->user_data);
  GetAllInputTensorShapes(context, node, inputs_shape);
  GetShapesPointers(inputs_shape, node->inputs->size, inputs_shape_ptr);
  GetAllInputTensorData(context, node, inputs_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConcatenationOutputTensor);

  reference_ops::Concatenation(data->params, inputs_shape_ptr, inputs_data,
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<DataType>(output));
}

}  // namespace

void* ConcatenationInitReference(TfLiteContext* context, const char* buffer,
                                 size_t length) {
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(ConcatenationOpData));
  if (data == nullptr) {
    return nullptr;
  }
  return new (data) ConcatenationOpData();
}

TfLiteStatus ConcatenationPrepareReference(TfLiteContext* context,
                                           TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  const TfLiteConcatenationParams* params =
      reinterpret_cast<const TfLiteConcatenationParams*>(node->builtin_data);

  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context,
                 num_inputs > 0 && num_inputs <= kConcatenationMaxInputNum);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input_tensor = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input_tensor != nullptr);
  TfLiteType input_type = input_tensor->type;
  TfLiteTensor* output_tensor =
      micro_context->AllocateTempOutputTensor(node, kConcatenationOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  micro_context->DeallocateTempTfLiteTensor(input_tensor);

  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteBool);
  TF_LITE_ENSURE_TYPES_EQ(context, output_type, input_type);

  ConcatenationOpData* data =
      static_cast<ConcatenationOpData*>(node->user_data);

  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, i);
    TF_LITE_ENSURE(context, input != nullptr);
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, input_type);
    int num_dimensions = NumDimensions(input);

    if (num_dimensions > RuntimeShape::kMaxSmallSize) {
      MicroPrintf(
          "Op Concatenation does not currently support num dimensions > %d "
          "Tensor has %d dimensions.",
          RuntimeShape::kMaxSmallSize, num_dimensions);
      return kTfLiteError;
    }

    if (input_type == kTfLiteInt8) {
      // Make sure there is no re-scaling needed for Int8 quantized kernel.
      TF_LITE_ENSURE_EQ(context, static_cast<double>(input->params.scale),
                        static_cast<double>(output_tensor->params.scale));
      TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                        output_tensor->params.zero_point);
    } else if (input_type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    }

#ifdef USE_TFLM_COMPRESSION
    data->scratch_indices[i] =
        micro_context->AllocateDecompressionScratchBuffer(node, i);
#endif  // USE_TFLM_COMPRESSION

    micro_context->DeallocateTempTfLiteTensor(input);
  }

  if (input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output_tensor->params.zero_point, 0);
  }

  data->params.axis = CalculatePositiveAxis(params->axis, output_tensor);
  data->params.inputs_count = node->inputs->size;

  micro_context->DeallocateTempTfLiteTensor(output_tensor);

  return kTfLiteOk;
}

TfLiteStatus ConcatenationEvalReference(TfLiteContext* context,
                                        TfLiteNode* node) {
  const TfLiteEvalTensor* output_tensor =
      tflite::micro::GetEvalOutput(context, node, kConcatenationOutputTensor);
  TfLiteType output_type = output_tensor->type;

  switch (output_type) {
    case kTfLiteFloat32:
      EvalUnquantized<float>(context, node);
      break;
    case kTfLiteInt32:
      EvalUnquantized<int32_t>(context, node);
      break;
    case kTfLiteInt8:
      EvalUnquantized<int8_t>(context, node);
      break;
    case kTfLiteInt64:
      EvalUnquantized<int64_t>(context, node);
      break;
    case kTfLiteInt16:
      EvalUnquantized<int16_t>(context, node);
      break;
    case kTfLiteBool:
      EvalUnquantized<bool>(context, node);
      break;
    default:
      MicroPrintf("Op Concatenation does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite
