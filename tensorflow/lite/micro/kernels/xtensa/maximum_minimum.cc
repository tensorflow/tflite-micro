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
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/reference/maximum_minimum.h"
#include "tensorflow/lite/micro/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace micro {
namespace {

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

inline constexpr int kMaxBroadcastDims = 8;

struct OpData {
  int input1_dims[kMaxBroadcastDims];
  int input2_dims[kMaxBroadcastDims];
  int output_extents[kMaxBroadcastDims];
#if defined(HIFI4) || defined(HIFI5)
  int input1_strides[kMaxBroadcastDims];
  int input2_strides[kMaxBroadcastDims];
#endif  // defined(HIFI4) || defined(HIFI5)
  int broadcast_dims;
  int output_flat_size;
  bool requires_broadcast;
  bool input1_matches_output;
  bool input2_matches_output;
};

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

void ExtendDims(const RuntimeShape& shape, int ndims, int* extended_dims) {
  const int dims_count = shape.DimensionsCount();
  TFLITE_DCHECK_GE(dims_count, 0);
  TFLITE_DCHECK_LE(dims_count, ndims);
  const int pad = ndims - dims_count;
  for (int i = 0; i < ndims; ++i) {
    extended_dims[i] = (i < pad) ? 1 : shape.Dims(i - pad);
  }
}

bool MatchesExtents(const int* dims, const int* extents, int ndims) {
  for (int i = 0; i < ndims; ++i) {
    if (dims[i] != extents[i]) {
      return false;
    }
  }
  return true;
}

#if defined(HIFI4) || defined(HIFI5)
void ComputeBroadcastStrides(const int* extended_dims,
                             const int* output_extents, int ndims,
                             int* strides) {
  int stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = (extended_dims[i] == output_extents[i]) ? stride : 0;
    stride *= extended_dims[i];
  }
}
#endif  // defined(HIFI4) || defined(HIFI5)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* op_data = context->AllocatePersistentBuffer(context, sizeof(OpData));
  return new (op_data) OpData();
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = static_cast<OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kMaximumMinimumInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kMaximumMinimumInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kMaximumMinimumOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const TfLiteType output_type = output->type;
  const RuntimeShape input1_shape = GetTensorShape(input1);
  const RuntimeShape input2_shape = GetTensorShape(input2);
  const RuntimeShape output_shape = GetTensorShape(output);

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);

  if (output_type != kTfLiteInt8) {
    return kTfLiteOk;
  }

  data->output_flat_size = output_shape.FlatSize();
  data->requires_broadcast = !(input1_shape == input2_shape);
  if (!data->requires_broadcast) {
    return kTfLiteOk;
  }

  const int max_dims =
      std::max({input1_shape.DimensionsCount(), input2_shape.DimensionsCount(),
                output_shape.DimensionsCount()});
  TF_LITE_ENSURE(context, max_dims >= 1 && max_dims <= kMaxBroadcastDims);

  // RuntimeShape in TFLM only supports up to kMaxSmallSize (6) dimensions, so
  // pad dimensions directly into OpData arrays rather than calling
  // RuntimeShape::ExtendedShape(8, ...).
  const int ndims = (max_dims <= 4) ? 4 : 8;
  data->broadcast_dims = ndims;
  ExtendDims(input1_shape, ndims, data->input1_dims);
  ExtendDims(input2_shape, ndims, data->input2_dims);
  ExtendDims(output_shape, ndims, data->output_extents);

  data->input1_matches_output =
      MatchesExtents(data->input1_dims, data->output_extents, ndims);
  data->input2_matches_output =
      MatchesExtents(data->input2_dims, data->output_extents, ndims);

#if defined(HIFI4) || defined(HIFI5)
  if (!data->input1_matches_output && !data->input2_matches_output) {
    ComputeBroadcastStrides(data->input1_dims, data->output_extents, ndims,
                            data->input1_strides);
    ComputeBroadcastStrides(data->input2_dims, data->output_extents, ndims,
                            data->input2_strides);
  }
#endif  // defined(HIFI4) || defined(HIFI5)

  return kTfLiteOk;
}

template <typename OpType>
TfLiteStatus HiFiEvalInt8(TfLiteNode* node, const OpContext& op_context) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *static_cast<const OpData*>(node->user_data);

  const int8_t* input1_data =
      tflite::micro::GetTensorData<int8_t>(op_context.input1);
  const int8_t* input2_data =
      tflite::micro::GetTensorData<int8_t>(op_context.input2);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(op_context.output);

  int err = 0;
  if (!data.requires_broadcast) {
    err = ExecElemKernelInt8<OpType>(output_data, input1_data, input2_data,
                                     data.output_flat_size);
    if (err != 0) {
      return kTfLiteError;
    }
    return kTfLiteOk;
  }

  if (!data.input1_matches_output && data.input2_matches_output) {
    // Broadcast input1 into output_data buffer, then execute element-wise op.
    err = xa_nn_broadcast_8_8(output_data, data.output_extents, input1_data,
                              data.input1_dims, data.broadcast_dims);
    if (err != 0) {
      return kTfLiteError;
    }

    err = ExecElemKernelInt8<OpType>(output_data, output_data, input2_data,
                                     data.output_flat_size);
    if (err != 0) {
      return kTfLiteError;
    }
  } else if (data.input1_matches_output && !data.input2_matches_output) {
    // Broadcast input2 into output_data buffer, then execute element-wise op.
    err = xa_nn_broadcast_8_8(output_data, data.output_extents, input2_data,
                              data.input2_dims, data.broadcast_dims);
    if (err != 0) {
      return kTfLiteError;
    }

    err = ExecElemKernelInt8<OpType>(output_data, input1_data, output_data,
                                     data.output_flat_size);
    if (err != 0) {
      return kTfLiteError;
    }
  } else {
    // Both inputs need broadcast.
#if defined(HIFI4) || defined(HIFI5)
    if (data.broadcast_dims == 4) {
      if constexpr (std::is_same<OpType, MinimumOp>::value) {
        err = xa_nn_elm_min_4D_Bcast_8x8_8(output_data, data.output_extents,
                                           input1_data, data.input1_strides,
                                           input2_data, data.input2_strides);
      } else if constexpr (std::is_same<OpType, MaximumOp>::value) {
        err = xa_nn_elm_max_4D_Bcast_8x8_8(output_data, data.output_extents,
                                           input1_data, data.input1_strides,
                                           input2_data, data.input2_strides);
      } else {
        return kTfLiteError;
      }
    } else if (data.broadcast_dims == 8) {
      if constexpr (std::is_same<OpType, MinimumOp>::value) {
        err = xa_nn_elm_min_8D_Bcast_8x8_8(output_data, data.output_extents,
                                           input1_data, data.input1_strides,
                                           input2_data, data.input2_strides);
      } else if constexpr (std::is_same<OpType, MaximumOp>::value) {
        err = xa_nn_elm_max_8D_Bcast_8x8_8(output_data, data.output_extents,
                                           input1_data, data.input1_strides,
                                           input2_data, data.input2_strides);
      } else {
        return kTfLiteError;
      }
    } else {
      return kTfLiteError;
    }
    if (err != 0) {
      return kTfLiteError;
    }
#else
    reference_ops::MaximumMinimumBroadcastSlow(
        tflite::micro::GetTensorShape(op_context.input1), input1_data,
        tflite::micro::GetTensorShape(op_context.input2), input2_data,
        tflite::micro::GetTensorShape(op_context.output), output_data,
        OpType::template op<int8_t>);
#endif  // defined(HIFI4) || defined(HIFI5)
  }

  return kTfLiteOk;
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
      return HiFiEvalInt8<OpType>(node, op_context);
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
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(Init, Prepare, Eval<MaximumOp>);
#else
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval<MaximumOp>);
#endif
}

TFLMRegistration Register_MINIMUM() {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return tflite::micro::RegisterOp(Init, Prepare, Eval<MinimumOp>);
#else
  return tflite::micro::RegisterOp(nullptr, nullptr, Eval<MinimumOp>);
#endif
}

}  // namespace micro
}  // namespace tflite
