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
#include "tensorflow/lite/micro/kernels/concatenation.h"

#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace micro {
namespace {

#if !defined(USE_TFLM_COMPRESSION) && \
    (defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
int ElementByteSize(TfLiteType type) {
  switch (type) {
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteInt16:
      return sizeof(int16_t);
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteInt64:
      return sizeof(int64_t);
    default:
      return 0;
  }
}

bool PrepareByteScaledShapesAndData(
    const TfLiteContext* context, const TfLiteNode* node,
    const TfLiteEvalTensor* output, int axis, int bytes_per_element,
    int32_t inputs_shape[kConcatenationMaxInputNum]
                        [RuntimeShape::kMaxSmallSize],
    const int32_t* inputs_shape_ptr[kConcatenationMaxInputNum],
    const int8_t* inputs_data[kConcatenationMaxInputNum],
    int32_t output_shape[RuntimeShape::kMaxSmallSize]) {
  const int num_inputs = node->inputs->size;
  const int num_out_dims = output->dims->size;
  if (bytes_per_element <= 0 || num_inputs <= 0 ||
      num_inputs > kConcatenationMaxInputNum || num_out_dims <= 0 ||
      num_out_dims > RuntimeShape::kMaxSmallSize || axis < 0 ||
      axis >= num_out_dims || output->data.raw == nullptr) {
    return false;
  }

  for (int d = 0; d < num_out_dims; ++d) {
    const int32_t dim = output->dims->data[d];
    if (dim <= 0) {
      return false;
    }
    output_shape[d] = (d == num_out_dims - 1) ? dim * bytes_per_element : dim;
  }

  int32_t concat_size = 0;
  for (int i = 0; i < num_inputs; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    if (t == nullptr || t->data.raw_const == nullptr ||
        t->dims->size != num_out_dims) {
      return false;
    }
    for (int d = 0; d < num_out_dims; ++d) {
      const int32_t dim = t->dims->data[d];
      if (dim <= 0) {
        return false;
      }
      inputs_shape[i][d] =
          (d == num_out_dims - 1) ? dim * bytes_per_element : dim;
      if (d != axis && inputs_shape[i][d] != output_shape[d]) {
        return false;
      }
    }
    concat_size += inputs_shape[i][axis];
    inputs_shape_ptr[i] = inputs_shape[i];
    inputs_data[i] = tflite::micro::GetTensorData<int8_t>(t);
  }

  return concat_size == output_shape[axis];
}

TfLiteStatus ConcatenationEvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConcatenationOutputTensor);
  const int bytes_per_element = ElementByteSize(output->type);
  if (bytes_per_element > 0) {
    const ConcatenationOpData* data =
        static_cast<const ConcatenationOpData*>(node->user_data);
    int32_t inputs_shape[kConcatenationMaxInputNum]
                        [RuntimeShape::kMaxSmallSize];
    const int32_t* inputs_shape_ptr[kConcatenationMaxInputNum];
    const int8_t* inputs_data[kConcatenationMaxInputNum];
    int32_t output_shape[RuntimeShape::kMaxSmallSize];

    if (PrepareByteScaledShapesAndData(
            context, node, output, data->params.axis, bytes_per_element,
            inputs_shape, inputs_shape_ptr, inputs_data, output_shape)) {
      const int num_out_dims = output->dims->size;
      const int err = xa_nn_concat_8_8(
          tflite::micro::GetTensorData<int8_t>(output), output_shape,
          inputs_data, inputs_shape_ptr, num_out_dims, node->inputs->size,
          num_out_dims, data->params.axis);
      if (err != 0) {
        return kTfLiteError;
      }
      return kTfLiteOk;
    }
  }
  return ConcatenationEvalReference(context, node);
}
#endif  // !defined(USE_TFLM_COMPRESSION) && (defined(HIFI3) || defined(HIFI4)
        // || defined(HIFI5) || defined(HIFI_IQ))

}  // namespace

TFLMRegistration Register_CONCATENATION() {
#if !defined(USE_TFLM_COMPRESSION) && \
    (defined(HIFI3) || defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  return tflite::micro::RegisterOp(ConcatenationInitReference,
                                   ConcatenationPrepareReference,
                                   ConcatenationEvalXtensa);
#else
  return tflite::micro::RegisterOp(ConcatenationInitReference,
                                   ConcatenationPrepareReference,
                                   ConcatenationEvalReference);
#endif
}

}  // namespace micro
}  // namespace tflite
