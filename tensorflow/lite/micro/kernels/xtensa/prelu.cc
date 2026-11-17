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

#include "tensorflow/lite/micro/kernels/prelu.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <new>

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/internal/types.h"
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

bool CanBroadcastToExtents(const int* dims, const int* extents, int ndims) {
  for (int i = 0; i < ndims; ++i) {
    if (dims[i] <= 0 || extents[i] <= 0) {
      return false;
    }
    if (dims[i] != extents[i] && dims[i] != 1) {
      return false;
    }
  }
  return true;
}
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

struct OpDataPreluXtensa {
  PreluParams reference_op_data;
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  int alpha_dims[kMaxBroadcastDims];
  int output_dims[kMaxBroadcastDims];
  int broadcast_dims;
  int output_flat_size;
  int scratch_tensor_index;
  bool requires_alpha_broadcast;
  bool use_nnlib;
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
};

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(OpDataPreluXtensa));
  if (data == nullptr) {
    return nullptr;
  }
  return new (data) OpDataPreluXtensa();
}

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  OpDataPreluXtensa* data = static_cast<OpDataPreluXtensa*>(node->user_data);

  TF_LITE_ENSURE_OK(
      context, PreparePreluParameters(context, node, &data->reference_op_data));

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  data->requires_alpha_broadcast = false;
  data->use_nnlib = false;

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPreluInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* alpha =
      micro_context->AllocateTempInputTensor(node, kPreluAlphaTensor);
  TF_LITE_ENSURE(context, alpha != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kPreluOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const TfLiteType output_type = output->type;
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape alpha_shape = GetTensorShape(alpha);
  const RuntimeShape output_shape = GetTensorShape(output);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(alpha);
  micro_context->DeallocateTempTfLiteTensor(output);

  if (output_type != kTfLiteInt8) {
    return kTfLiteOk;
  }

  const int flat_size = output_shape.FlatSize();
  const PreluParams& params = data->reference_op_data;
  if (flat_size <= 0 || params.input_offset < -127 ||
      params.input_offset > 128 || params.alpha_offset < -127 ||
      params.alpha_offset > 128 || params.output_offset < -128 ||
      params.output_offset > 127 || params.output_shift_1 < -31 ||
      params.output_shift_1 > 31 || params.output_shift_2 < -31 ||
      params.output_shift_2 > 31) {
    return kTfLiteOk;
  }

  if (input_shape == output_shape && alpha_shape == output_shape) {
    data->output_flat_size = flat_size;
    data->use_nnlib = true;
    return kTfLiteOk;
  }

  const int max_dims =
      std::max({input_shape.DimensionsCount(), alpha_shape.DimensionsCount(),
                output_shape.DimensionsCount()});
  if (max_dims < 1 || max_dims > kMaxBroadcastDims) {
    return kTfLiteOk;
  }

  int extended_input_dims[kMaxBroadcastDims];
  ExtendDims(input_shape, max_dims, extended_input_dims);
  ExtendDims(alpha_shape, max_dims, data->alpha_dims);
  ExtendDims(output_shape, max_dims, data->output_dims);

  if (!MatchesExtents(extended_input_dims, data->output_dims, max_dims) ||
      !CanBroadcastToExtents(data->alpha_dims, data->output_dims, max_dims)) {
    return kTfLiteOk;
  }

  if (!MatchesExtents(data->alpha_dims, data->output_dims, max_dims)) {
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, flat_size * sizeof(int8_t),
                                   &data->scratch_tensor_index));
    data->broadcast_dims = max_dims;
    data->requires_alpha_broadcast = true;
  }

  data->output_flat_size = flat_size;
  data->use_nnlib = true;
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

  return kTfLiteOk;
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataPreluXtensa& xtensa_data =
      *static_cast<const OpDataPreluXtensa*>(node->user_data);
  const PreluParams& params = xtensa_data.reference_op_data;

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kPreluInputTensor);
  const TfLiteEvalTensor* alpha =
      tflite::micro::GetEvalInput(context, node, kPreluAlphaTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kPreluOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32: {
      PreluEvalReference<float>(params, input, alpha, output);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      if (xtensa_data.use_nnlib) {
        const int8_t* alpha_data = tflite::micro::GetTensorData<int8_t>(alpha);
        if (xtensa_data.requires_alpha_broadcast) {
          int8_t* alpha_broadcasted =
              static_cast<int8_t*>(context->GetScratchBuffer(
                  context, xtensa_data.scratch_tensor_index));
          if (alpha_broadcasted == nullptr) {
            return kTfLiteError;
          }
          int err = xa_nn_broadcast_8_8(
              alpha_broadcasted, xtensa_data.output_dims, alpha_data,
              xtensa_data.alpha_dims, xtensa_data.broadcast_dims);
          if (err != 0) {
            return kTfLiteError;
          }
          alpha_data = alpha_broadcasted;
        }

        int err = xa_nn_vec_prelu_asym8s_asym8s(
            tflite::micro::GetTensorData<int8_t>(output),
            tflite::micro::GetTensorData<int8_t>(input), alpha_data,
            params.input_offset, params.alpha_offset,
            params.output_multiplier_2, params.output_shift_2,
            params.output_multiplier_1, params.output_shift_1,
            params.output_offset, xtensa_data.output_flat_size);
        if (err != 0) {
          return kTfLiteError;
        }
        return kTfLiteOk;
      }
#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      PreluEvalReference<int8_t>(params, input, alpha, output);
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      PreluEvalReference<int16_t>(params, input, alpha, output);
      return kTfLiteOk;
    }
    default: {
      MicroPrintf("Input type '%s' is not supported.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }
}

}  // namespace

TFLMRegistration Register_PRELU() {
  return tflite::micro::RegisterOp(PreluInit, PreluPrepare, PreluEval);
}

}  // namespace micro
}  // namespace tflite
