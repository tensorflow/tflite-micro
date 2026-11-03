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

#include "tensorflow/lite/micro/kernels/maximum_minimum.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/reference/maximum_minimum.h"
#include "tensorflow/lite/micro/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

template <typename OpType>
int ExecElemKernelInt8(int8_t* out_data, const int8_t* in1_data,
                       const int8_t* in2_data, int num_elements) {
  if constexpr (std::is_same<OpType, MinimumOp>::value) {
    return xa_nn_elm_min_8x8_8(out_data, in1_data, in2_data, num_elements);
  } else if constexpr (std::is_same<OpType, MaximumOp>::value) {
    return xa_nn_elm_max_8x8_8(out_data, in1_data, in2_data, num_elements);
  } else {
    return -1;
  }
}

template <int NDims>
void ExtendDims(const RuntimeShape& shape, int (&extended_dims)[NDims]) {
  const int dims_count = shape.DimensionsCount();
  TFLITE_DCHECK_GE(dims_count, 0);
  TFLITE_DCHECK_LE(dims_count, NDims);
  const int pad = NDims - dims_count;
  for (int i = 0; i < NDims; ++i) {
    extended_dims[i] = (i < pad) ? 1 : shape.Dims(i - pad);
  }
}

template <int NDims>
bool MatchesExtents(const int (&dims)[NDims], const int (&extents)[NDims]) {
  for (int i = 0; i < NDims; ++i) {
    if (dims[i] != extents[i]) {
      return false;
    }
  }
  return true;
}

#if defined(HIFI4) || defined(HIFI5)
template <int NDims>
void ComputeBroadcastStrides(const int (&extended_dims)[NDims],
                             const int (&output_extents)[NDims],
                             int (&strides)[NDims]) {
  int stride = 1;
  for (int i = NDims - 1; i >= 0; --i) {
    strides[i] = (extended_dims[i] == output_extents[i]) ? stride : 0;
    stride *= extended_dims[i];
  }
}
#endif  // defined(HIFI4) || defined(HIFI5)

template <typename OpType, int NDims>
TfLiteStatus MaximumMinimumBroadcastInt8(
    TfLiteContext* context, const RuntimeShape& unextended_input1_shape,
    const int8_t* input1_data, const RuntimeShape& unextended_input2_shape,
    const int8_t* input2_data, const RuntimeShape& unextended_output_shape,
    int8_t* output_data) {
  int err = 0;

  // RuntimeShape in TFLM only supports up to kMaxSmallSize (6) dimensions, so
  // pad dimensions directly into int arrays rather than calling
  // RuntimeShape::ExtendedShape(8, ...).
  int input1_dims[NDims];
  int input2_dims[NDims];
  int output_extents[NDims];
  ExtendDims<NDims>(unextended_input1_shape, input1_dims);
  ExtendDims<NDims>(unextended_input2_shape, input2_dims);
  ExtendDims<NDims>(unextended_output_shape, output_extents);

  const bool input1_matches_output =
      MatchesExtents<NDims>(input1_dims, output_extents);
  const bool input2_matches_output =
      MatchesExtents<NDims>(input2_dims, output_extents);
  const int output_flat_size = unextended_output_shape.FlatSize();

  if (!input1_matches_output && input2_matches_output) {
    // Broadcast input1 into output_data buffer, then execute element-wise op.
    err = xa_nn_broadcast_8_8(output_data, output_extents, input1_data,
                              input1_dims, NDims);
    TF_LITE_ENSURE_EQ(context, err, 0);

    err = ExecElemKernelInt8<OpType>(output_data, output_data, input2_data,
                                     output_flat_size);
    TF_LITE_ENSURE_EQ(context, err, 0);
  } else if (input1_matches_output && !input2_matches_output) {
    // Broadcast input2 into output_data buffer, then execute element-wise op.
    err = xa_nn_broadcast_8_8(output_data, output_extents, input2_data,
                              input2_dims, NDims);
    TF_LITE_ENSURE_EQ(context, err, 0);

    err = ExecElemKernelInt8<OpType>(output_data, input1_data, output_data,
                                     output_flat_size);
    TF_LITE_ENSURE_EQ(context, err, 0);
  } else {
    // Both inputs need broadcast.
#if defined(HIFI4) || defined(HIFI5)
    int input1_strides[NDims];
    int input2_strides[NDims];
    ComputeBroadcastStrides<NDims>(input1_dims, output_extents, input1_strides);
    ComputeBroadcastStrides<NDims>(input2_dims, output_extents, input2_strides);

    if constexpr (NDims == 4) {
      if constexpr (std::is_same<OpType, MinimumOp>::value) {
        err = xa_nn_elm_min_4D_Bcast_8x8_8(output_data, output_extents,
                                           input1_data, input1_strides,
                                           input2_data, input2_strides);
      } else if constexpr (std::is_same<OpType, MaximumOp>::value) {
        err = xa_nn_elm_max_4D_Bcast_8x8_8(output_data, output_extents,
                                           input1_data, input1_strides,
                                           input2_data, input2_strides);
      } else {
        return kTfLiteError;
      }
    } else if constexpr (NDims == 8) {
      if constexpr (std::is_same<OpType, MinimumOp>::value) {
        err = xa_nn_elm_min_8D_Bcast_8x8_8(output_data, output_extents,
                                           input1_data, input1_strides,
                                           input2_data, input2_strides);
      } else if constexpr (std::is_same<OpType, MaximumOp>::value) {
        err = xa_nn_elm_max_8D_Bcast_8x8_8(output_data, output_extents,
                                           input1_data, input1_strides,
                                           input2_data, input2_strides);
      } else {
        return kTfLiteError;
      }
    } else {
      return kTfLiteError;
    }
    TF_LITE_ENSURE_EQ(context, err, 0);
#else
    reference_ops::MaximumMinimumBroadcastSlow(
        unextended_input1_shape, input1_data, unextended_input2_shape,
        input2_data, unextended_output_shape, output_data,
        OpType::template op<int8_t>);
#endif  // defined(HIFI4) || defined(HIFI5)
  }

  return kTfLiteOk;
}

template <typename OpType>
TfLiteStatus HiFiEvalInt8(TfLiteContext* context, const OpContext& op_context) {
  const RuntimeShape unextended_input1_shape =
      tflite::micro::GetTensorShape(op_context.input1);
  const RuntimeShape unextended_input2_shape =
      tflite::micro::GetTensorShape(op_context.input2);
  const RuntimeShape unextended_output_shape =
      tflite::micro::GetTensorShape(op_context.output);

  const int8_t* input1_data =
      tflite::micro::GetTensorData<int8_t>(op_context.input1);
  const int8_t* input2_data =
      tflite::micro::GetTensorData<int8_t>(op_context.input2);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(op_context.output);

  if (unextended_input1_shape == unextended_input2_shape) {
    int err = ExecElemKernelInt8<OpType>(output_data, input1_data, input2_data,
                                         unextended_output_shape.FlatSize());
    TF_LITE_ENSURE_EQ(context, err, 0);
    return kTfLiteOk;
  }

  const int max_dims = std::max({unextended_input1_shape.DimensionsCount(),
                                 unextended_input2_shape.DimensionsCount(),
                                 unextended_output_shape.DimensionsCount()});
  TF_LITE_ENSURE(context, max_dims >= 1 && max_dims <= 8);

  if (max_dims <= 4) {
    return MaximumMinimumBroadcastInt8<OpType, 4>(
        context, unextended_input1_shape, input1_data, unextended_input2_shape,
        input2_data, unextended_output_shape, output_data);
  }
  return MaximumMinimumBroadcastInt8<OpType, 8>(
      context, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

template <typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

  switch (op_context.output->type) {
    case kTfLiteFloat32:
      TFLiteOperation<float, OpType>(context, node, op_context);
      break;
    case kTfLiteInt8:
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      return HiFiEvalInt8<OpType>(context, op_context);
#else
      TFLiteOperation<int8_t, OpType>(context, node, op_context);
      break;
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
    case kTfLiteInt16:
      TFLiteOperation<int16_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt32:
      TFLiteOperation<int32_t, OpType>(context, node, op_context);
      break;
    case kTfLiteInt64:
      TFLiteOperation<int64_t, OpType>(context, node, op_context);
      break;
    default:
      MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                  TfLiteTypeGetName(op_context.output->type),
                  op_context.output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MAXIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval<MaximumOp>);
}

TFLMRegistration Register_MINIMUM() {
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval<MinimumOp>);
}

}  // namespace micro
}  // namespace tflite
