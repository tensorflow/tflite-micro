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

#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_reduce.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace micro {
namespace {

constexpr int kMaxSupportedDimsHifi = 4;

bool HasPositiveDims(const TfLiteIntArray* dims) {
  for (int i = 0; i < dims->size; ++i) {
    if (dims->data[i] <= 0) {
      return false;
    }
  }
  return true;
}

bool CanUseHifiReduce(const TfLiteTensor* input, const TfLiteTensor* output,
                      const TfLiteTensor* axis, XtensaReduceOpData* xt_data) {
  if (input->dims->size < 1 || input->dims->size > kMaxSupportedDimsHifi ||
      output->dims->size < 0 || output->dims->size > kMaxSupportedDimsHifi) {
    return false;
  }
  if (!HasPositiveDims(input->dims) || !HasPositiveDims(output->dims)) {
    return false;
  }
  if (!IsConstantTensor(axis)) {
    return false;
  }

  const int num_axis = ElementCount(*axis->dims);
  int num_resolved_axis = 0;
  if (!reference_ops::ResolveAxis(
          input->dims->size, tflite::micro::GetTensorData<int>(axis), num_axis,
          xt_data->resolved_axis, &num_resolved_axis)) {
    return false;
  }
  if (num_resolved_axis <= 0 || num_resolved_axis > kMaxSupportedDimsHifi) {
    return false;
  }
  xt_data->num_resolved_axis = num_resolved_axis;
  return true;
}

}  // namespace

TfLiteStatus ReducePrepareMaxHifi(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* xt_data = static_cast<XtensaReduceOpData*>(node->user_data);
  xt_data->is_hifi_supported = false;

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kReduceInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kReduceOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* axis =
      micro_context->AllocateTempInputTensor(node, kReduceAxisTensor);
  TF_LITE_ENSURE(context, axis != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  if (input->type == kTfLiteInt8) {
    const OpDataReduce& op_data = xt_data->reference_op_data;
    TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data.input_scale),
                      static_cast<double>(op_data.output_scale));
    TF_LITE_ENSURE_EQ(context, op_data.input_zp, op_data.output_zp);
  }

  TfLiteStatus status = kTfLiteOk;
  if (input->type == kTfLiteInt8 &&
      CanUseHifiReduce(input, output, axis, xt_data)) {
    const int required_scratch = xa_nn_reduce_getsize_nhwc(
        PREC_ASYM8S, input->dims->data, input->dims->size,
        xt_data->resolved_axis, xt_data->num_resolved_axis, REDUCE_MAX);
    if (required_scratch <= 0) {
      MicroPrintf("reduce_max_4D_asym8: xa_nn_reduce_getsize_nhwc failed");
      status = kTfLiteError;
    } else {
      status = context->RequestScratchBufferInArena(
          context, required_scratch, &xt_data->scratch_tensor_index);
      if (status == kTfLiteOk) {
        xt_data->is_hifi_supported = true;
      }
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return status;
}

TfLiteStatus ReducePrepareMeanHifi(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* xt_data = static_cast<XtensaReduceOpData*>(node->user_data);
  const OpDataReduce& op_data = xt_data->reference_op_data;
  xt_data->is_hifi_supported = false;

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kReduceInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kReduceOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* axis =
      micro_context->AllocateTempInputTensor(node, kReduceAxisTensor);
  TF_LITE_ENSURE(context, axis != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  TfLiteStatus status = kTfLiteOk;
  if ((input->type == kTfLiteInt8 || input->type == kTfLiteInt16) &&
      CanUseHifiReduce(input, output, axis, xt_data)) {
    int64_t num_elements_in_axis = 0;
    if (reduce_utils::CheckedReducedElementCount(
            input->dims->data, input->dims->size, xt_data->resolved_axis,
            xt_data->num_resolved_axis, num_elements_in_axis) &&
        num_elements_in_axis > 0) {
      int shift =
          63 - CountLeadingZeros(static_cast<uint64_t>(num_elements_in_axis));
      shift = std::min(shift, 32);
      shift = std::min(shift, 31 + op_data.shift);
      xt_data->updated_multiplier = static_cast<int32_t>(
          (static_cast<int64_t>(op_data.multiplier) << shift) /
          num_elements_in_axis);
      xt_data->updated_shift = op_data.shift - shift;

      // Guard NNLib parameter bounds and the rare case where updated_multiplier
      // == 0x40000000 && updated_shift == 1 with non-zero matching zero points
      // across >1 reduced elements (where NNLib's same_quant shortcut omits
      // zero-point adjustment).
      const bool valid_nnlib_quant =
          xt_data->updated_multiplier >= 0 && xt_data->updated_shift >= -31 &&
          xt_data->updated_shift <= 31 &&
          !(num_elements_in_axis > 1 && op_data.input_zp != 0 &&
            op_data.input_zp == op_data.output_zp &&
            xt_data->updated_multiplier == 0x40000000 &&
            xt_data->updated_shift == 1);

      if (valid_nnlib_quant) {
        const int inp_precision =
            (input->type == kTfLiteInt16) ? PREC_ASYM16S : PREC_ASYM8S;
        const int required_scratch = xa_nn_reduce_getsize_nhwc(
            inp_precision, input->dims->data, input->dims->size,
            xt_data->resolved_axis, xt_data->num_resolved_axis, REDUCE_MEAN);
        if (required_scratch <= 0) {
          MicroPrintf("reduce_mean_4D: xa_nn_reduce_getsize_nhwc failed");
          status = kTfLiteError;
        } else {
          status = context->RequestScratchBufferInArena(
              context, required_scratch, &xt_data->scratch_tensor_index);
          if (status == kTfLiteOk) {
            xt_data->is_hifi_supported = true;
          }
        }
      }
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return status;
}

TfLiteStatus ReduceEvalMeanHifi(TfLiteContext* context, TfLiteNode* node,
                                const XtensaReduceOpData& data) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kReduceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kReduceOutputTensor);

  void* p_scratch =
      context->GetScratchBuffer(context, data.scratch_tensor_index);
  if (p_scratch == nullptr) {
    return kTfLiteError;
  }

  const int scalar_out_dims[1] = {1};
  const int num_out_dims = (output->dims->size == 0) ? 1 : output->dims->size;
  const int* out_dims_ptr =
      (output->dims->size == 0) ? scalar_out_dims : output->dims->data;
  const OpDataReduce& op_data = data.reference_op_data;

  int err = 0;
  if (input->type == kTfLiteInt8) {
    err = xa_nn_reduce_mean_4D_asym8s_asym8s(
        tflite::micro::GetTensorData<int8_t>(output), out_dims_ptr,
        tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
        data.resolved_axis, num_out_dims, input->dims->size,
        data.num_resolved_axis, op_data.input_zp, data.updated_multiplier,
        data.updated_shift, op_data.output_zp, p_scratch);
  } else if (input->type == kTfLiteInt16) {
    err = xa_nn_reduce_mean_4D_asym16s_asym16s(
        tflite::micro::GetTensorData<int16_t>(output), out_dims_ptr,
        tflite::micro::GetTensorData<int16_t>(input), input->dims->data,
        data.resolved_axis, num_out_dims, input->dims->size,
        data.num_resolved_axis, op_data.input_zp, data.updated_multiplier,
        data.updated_shift, op_data.output_zp, p_scratch);
  } else {
    return kTfLiteError;
  }

  if (err != 0) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ReduceEvalMaxHifi(TfLiteContext* context, TfLiteNode* node,
                               const XtensaReduceOpData& data) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kReduceInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kReduceOutputTensor);

  void* p_scratch =
      context->GetScratchBuffer(context, data.scratch_tensor_index);
  if (p_scratch == nullptr) {
    return kTfLiteError;
  }

  const int scalar_out_dims[1] = {1};
  const int num_out_dims = (output->dims->size == 0) ? 1 : output->dims->size;
  const int* out_dims_ptr =
      (output->dims->size == 0) ? scalar_out_dims : output->dims->data;

  int err = xa_nn_reduce_max_4D_asym8s_asym8s(
      tflite::micro::GetTensorData<int8_t>(output), out_dims_ptr,
      tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
      data.resolved_axis, num_out_dims, input->dims->size,
      data.num_resolved_axis, p_scratch);
  if (err != 0) {
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace micro
}  // namespace tflite

#endif  // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
