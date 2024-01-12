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

#if defined(VISION_P6)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_reduce.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

inline void OperandDims4D(uint32_t* dims, TfLiteTensor* opnd) {
  for (int i = NumDimensions(opnd) - 1, j = 0; i >= 0; i--, j++) {
    dims[j] = SizeOfDimension(opnd, i);
  }
  return;
}

// This function is duplicated from reference/reduce.h
// This method parses the input 'axis' to remove duplicates and handle negative
// values, and returns a valid 'out_axis'
inline bool ResolveAxis(const int num_dims, const int* axis,
                        const int64_t num_axis, int* out_axis,
                        int* out_num_axis) {
  *out_num_axis = 0;  // Just in case.
  // Short-circuit axis resolution for scalars; the axis will go unused.
  if (num_dims == 0) {
    return true;
  }
  // o(n^2) is fine since out_num_axis should be really small, mostly <= 4
  for (int64_t idx = 0; idx < num_axis; ++idx) {
    // Handle negative index. A positive index 'p_idx' can be represented as a
    // negative index 'n_idx' as: n_idx = p_idx-num_dims
    // eg: For num_dims=3, [0, 1, 2] is the same as [-3, -2, -1]  */
    int current = axis[idx] < 0 ? (axis[idx] + num_dims) : axis[idx];
    TFLITE_DCHECK(current >= 0 && current < num_dims);
    if (current < 0 || current >= num_dims) {
      return false;
    }
    bool is_dup = false;
    for (int j = 0; j < *out_num_axis; ++j) {
      if (out_axis[j] == current) {
        is_dup = true;
        break;
      }
    }
    if (!is_dup) {
      out_axis[*out_num_axis] = current;
      *out_num_axis += 1;
    }
  }
  return true;
}

TfLiteStatus ReducePrepareVision(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  XtensaReduceOpData* data =
      reinterpret_cast<XtensaReduceOpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);

  uint32_t input_dims[4] = {1, 1, 1, 1};
  uint32_t output_dims[4] = {1, 1, 1, 1};
  uint32_t should_reduce_r[4] = {0, 0, 0, 0};
  int32_t resolved_axis[4] = {0, 0, 0, 0};
  OperandDims4D(input_dims, input);
  OperandDims4D(output_dims, output);

  const int input_rank = NumDimensions(input);
  // Interpret an axis tensor with null dimensions as a scalar
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  // Resolve axis.
  int num_resolved_axis = 0;
  if (!ResolveAxis(input->dims->size, axis->data.i32, num_axis, resolved_axis,
                   &num_resolved_axis)) {
    return kTfLiteError;
  }

  // ResolveAxis should eliminate dupes and negative axis, so the number of axis
  // should be no greater than the input rank.
  TFLITE_DCHECK(num_resolved_axis <= input_rank);

  bool should_reduce[4] = {false, false, false, false};

  for (int32_t i = 0; i < num_resolved_axis; ++i) {
    int32_t axis_d = resolved_axis[i];
    should_reduce[axis_d] = true;
  }

  // reverse axes and align it to dimension 0 as OperandDims4D
  for (int axis_i = 0; axis_i < input_rank; ++axis_i) {
    should_reduce_r[input_rank - 1 - axis_i] =
        static_cast<uint32_t>(should_reduce[axis_i]);
  }

  uint32_t context_size = 0;
  uint32_t status = xiReduceGetMemReqd_Context(&context_size);
  if (!status && context_size) {
    void* context_data =
        context->AllocatePersistentBuffer(context, context_size);
    if (context_data == nullptr) {
      return kTfLiteError;
    }
    data->p_context = reinterpret_cast<uint8_t*>(context_data);
    data->context_size = context_size;
  }

  status = xiReduceSetContext(data->p_context, data->context_size, input_dims,
                              output_dims, should_reduce_r);

  if (status) {
    return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus ReduceEvalVision(const XtensaReduceOpData& data,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  const uint32_t input_size = NumElements(input->dims);
  const uint32_t output_size = NumElements(output->dims);

  xiReduce(data.p_context, data.context_size,
           const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
           input_size, tflite::micro::GetTensorData<int8_t>(output),
           output_size);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISION_P6)
