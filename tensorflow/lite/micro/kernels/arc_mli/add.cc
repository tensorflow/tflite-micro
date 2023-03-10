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

#include "tensorflow/lite/kernels/internal/reference/add.h"

#include <algorithm>
#include <limits>

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  bool requires_broadcast;

  // These fields are used in both the general 8-bit -> 8bit quantized path,
  // and the special 16-bit -> 16bit quantized path
  int input1_shift;
  int input2_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;

  // These fields are used only in the general 8-bit -> 8bit quantized path
  int32_t input1_multiplier;
  int32_t input2_multiplier;
  int32_t output_multiplier;
  int output_shift;
  int left_shift;
  int32_t input1_offset;
  int32_t input2_offset;
  int32_t output_offset;

  // Used only for float evals:
  float output_activation_min_f32;
  float output_activation_max_f32;

  // The result of checking if MLI optimized version of tensors can be used.
  bool is_mli_applicable;

  // Tensors in MLI format.
  mutable ops::micro::MliTensorInterface mli_input1;
  mutable ops::micro::MliTensorInterface mli_input2;
  mutable ops::micro::MliTensorInterface mli_out;
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteAddParams* params,
                             const TfLiteTensor* input1,
                             const TfLiteTensor* input2, TfLiteTensor* output,
                             OpData* data) {
  data->requires_broadcast = !HaveSameShapes(input1, input2);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));

    // MLI 2.0 optimized version only supports int8_t datatype and min/max
    // within container range. Broadcasting isn't supported on the primitive
    // level (but might be implemented as part of slicing in future)
#ifdef MLI_2_0  //
    data->is_mli_applicable =
        (input1->type == kTfLiteInt8) && (input2->type == kTfLiteInt8) &&
        (output->type == kTfLiteInt8) && !data->requires_broadcast &&
        data->output_activation_min == std::numeric_limits<int8_t>::min() &&
        data->output_activation_max == std::numeric_limits<int8_t>::max();
#else
    data->is_mli_applicable = false;
#endif

    if (data->is_mli_applicable) {
      data->mli_input1 =
          ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
              context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
      data->mli_input2 =
          ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
              context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
      data->mli_out = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
          context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));

      ops::micro::ConvertToMliTensor(input1, &data->mli_input1);
      ops::micro::ConvertToMliTensor(input2, &data->mli_input2);
      ops::micro::ConvertToMliTensor(output, &data->mli_out);
      /* Flatten tensors to simplify the process (as we don't support
       * broadcasting). */
      data->mli_input1.Shape()[0] =
          mli_hlp_count_elem_num(data->mli_input1.MliTensor(), 0);
      data->mli_input2.Shape()[0] =
          mli_hlp_count_elem_num(data->mli_input2.MliTensor(), 0);
      data->mli_out.Shape()[0] =
          mli_hlp_count_elem_num(data->mli_out.MliTensor(), 0);
      data->mli_input1.MemStride()[0] = data->mli_input2.MemStride()[0] = 1;
      data->mli_out.MemStride()[0] = 1;
      *data->mli_input1.Rank() = *data->mli_input2.Rank() = 1;
      *data->mli_out.Rank() = 1;
    }
  } else {
    data->is_mli_applicable = false;
  }

#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    // 8bit -> 8bit general quantized path, with general rescalings
    data->input1_offset = -input1->params.zero_point;
    data->input2_offset = -input2->params.zero_point;
    data->output_offset = output->params.zero_point;
    data->left_shift = (output->type == kTfLiteInt16) ? 15 : 20;
    const double twice_max_input_scale =
        2 * static_cast<double>(
                std::max(input1->params.scale, input2->params.scale));
    const double real_input1_multiplier =
        static_cast<double>(input1->params.scale) / twice_max_input_scale;
    const double real_input2_multiplier =
        static_cast<double>(input2->params.scale) / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale /
        ((1 << data->left_shift) * static_cast<double>(output->params.scale));

    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &data->input1_multiplier, &data->input1_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_input2_multiplier, &data->input2_multiplier, &data->input2_shift);

    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &data->output_multiplier, &data->output_shift);

    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  } else if (output->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation,
                             &data->output_activation_min_f32,
                             &data->output_activation_max_f32);
#endif  // !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  }

  return kTfLiteOk;
}

TfLiteStatus EvalAdd(TfLiteContext* context, TfLiteNode* node,
                     TfLiteAddParams* params, const OpData* data,
                     const TfLiteEvalTensor* input1,
                     const TfLiteEvalTensor* input2, TfLiteEvalTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  tflite::ArithmeticParams op_params;
  SetActivationParams(data->output_activation_min_f32,
                      data->output_activation_max_f32, &op_params);
  if (data->requires_broadcast) {
    reference_ops::BroadcastAdd4DSlow(
        op_params, tflite::micro::GetTensorShape(input1),
        tflite::micro::GetTensorData<float>(input1),
        tflite::micro::GetTensorShape(input2),
        tflite::micro::GetTensorData<float>(input2),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<float>(output));
  } else {
    reference_ops::Add(op_params, tflite::micro::GetTensorShape(input1),
                       tflite::micro::GetTensorData<float>(input1),
                       tflite::micro::GetTensorShape(input2),
                       tflite::micro::GetTensorData<float>(input2),
                       tflite::micro::GetTensorShape(output),
                       tflite::micro::GetTensorData<float>(output));
  }
  return kTfLiteOk;
#else
  MicroPrintf("Node configuration is not supported by ARC MLI Library.");
  return kTfLiteError;
#endif
}

TfLiteStatus EvalAddQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteAddParams* params, const OpData* data,
                              const TfLiteEvalTensor* input1,
                              const TfLiteEvalTensor* input2,
                              TfLiteEvalTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  tflite::ArithmeticParams op_params;
  op_params.left_shift = data->left_shift;
  op_params.input1_offset = data->input1_offset;
  op_params.input1_multiplier = data->input1_multiplier;
  op_params.input1_shift = data->input1_shift;
  op_params.input2_offset = data->input2_offset;
  op_params.input2_multiplier = data->input2_multiplier;
  op_params.input2_shift = data->input2_shift;
  op_params.output_offset = data->output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  SetActivationParams(data->output_activation_min, data->output_activation_max,
                      &op_params);
  bool need_broadcast = reference_ops::ProcessBroadcastShapes(
      tflite::micro::GetTensorShape(input1),
      tflite::micro::GetTensorShape(input2), &op_params);

  switch (output->type) {
    case kTfLiteInt8: {
      if (need_broadcast) {
        reference_integer_ops::BroadcastAdd4DSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
        reference_integer_ops::Add(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int8_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int8_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      }
      break;
    }
    case kTfLiteInt16: {
      if (need_broadcast) {
        reference_ops::BroadcastAdd4DSlow(
            op_params, tflite::micro::GetTensorShape(input1),
            tflite::micro::GetTensorData<int16_t>(input1),
            tflite::micro::GetTensorShape(input2),
            tflite::micro::GetTensorData<int16_t>(input2),
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output));
      } else {
        reference_ops::Add(op_params, tflite::micro::GetTensorShape(input1),
                           tflite::micro::GetTensorData<int16_t>(input1),
                           tflite::micro::GetTensorShape(input2),
                           tflite::micro::GetTensorData<int16_t>(input2),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<int16_t>(output),
                           false);
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.",
                  TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
#else
  MicroPrintf("Node configuration is not supported by ARC MLI Library.");
  return kTfLiteError;
#endif
}

TfLiteStatus EvalMLIAddInt8(TfLiteContext* context, TfLiteNode* node,
                            TfLiteAddParams* params, const OpData* data,
                            const TfLiteEvalTensor* input1,
                            const TfLiteEvalTensor* input2,
                            TfLiteEvalTensor* output) {
#ifdef MLI_2_0
  TF_LITE_ENSURE(context, data->is_mli_applicable == true);
  TF_LITE_ENSURE(context, input1->type == kTfLiteInt8);
  TF_LITE_ENSURE(context, input2->type == kTfLiteInt8);
  TF_LITE_ENSURE(context, output->type == kTfLiteInt8);

  ops::micro::MliTensorAttachBuffer<int8_t>(input1, &data->mli_input1);
  ops::micro::MliTensorAttachBuffer<int8_t>(input2, &data->mli_input2);
  ops::micro::MliTensorAttachBuffer<int8_t>(output, &data->mli_out);

  // mli_mov config and tensors for data in fast (local) memory with interface
  mli_mov_cfg_t copy_config;
  mli_mov_cfg_for_copy(&copy_config);
  mli_tensor input1_local_tsr = *data->mli_input1.MliTensor();
  mli_tensor input2_local_tsr = *data->mli_input2.MliTensor();
  mli_tensor out_local_tsr = *data->mli_out.MliTensor();
  ops::micro::MliTensorInterface input1_local(&input1_local_tsr);
  ops::micro::MliTensorInterface input2_local(&input2_local_tsr);
  ops::micro::MliTensorInterface out_local(&out_local_tsr);

  /* allocate the local buffers, and compute the slice size */
  TF_LITE_ENSURE_STATUS(ops::micro::get_arc_scratch_buffer_for_eltwise_tensors(
      context, &input1_local, &input2_local, &out_local));
  TF_LITE_ENSURE(context, *input1_local.Rank() == 1 &&
                              *input2_local.Rank() == 1 &&
                              *out_local.Rank() == 1);
  uint32_t min_capacity = *input1_local.DataCapacity();
  min_capacity = std::min(min_capacity, *input2_local.DataCapacity());
  min_capacity = std::min(min_capacity, *out_local.DataCapacity());
  const int slice_dim = 0;
  const int slice_size =
      min_capacity / mli_hlp_tensor_element_size(out_local.MliTensor());

  /* is_local indicates that the tensor is already in local memory,
     so in that case the original tensor can be used,
     and there is no need to copy it to the local tensor*/
  const bool input1_is_local =
      input1_local.Data<int8_t>() == data->mli_input1.Data<int8_t>();
  const bool input2_is_local =
      input2_local.Data<int8_t>() == data->mli_input2.Data<int8_t>();
  const bool out_is_local =
      out_local.Data<int8_t>() == data->mli_out.Data<int8_t>();

  ops::micro::TensorSlicer input1_slice(data->mli_input1.MliTensor(), slice_dim,
                                        slice_size);
  ops::micro::TensorSlicer input2_slice(data->mli_input2.MliTensor(), slice_dim,
                                        slice_size);
  ops::micro::TensorSlicer out_slice(data->mli_out.MliTensor(), slice_dim,
                                     slice_size);

  mli_tensor* input1_tsr =
      input1_is_local ? input1_slice.Sub() : input1_local.MliTensor();
  mli_tensor* input2_tsr =
      input2_is_local ? input2_slice.Sub() : input2_local.MliTensor();
  mli_tensor* out_tsr = out_is_local ? out_slice.Sub() : out_local.MliTensor();

  while (!out_slice.Done()) {
    mli_mov_tensor_sync(input1_slice.Sub(), &copy_config, input1_tsr);
    mli_mov_tensor_sync(input2_slice.Sub(), &copy_config, input2_tsr);

    mli_krn_eltwise_add_sa8(input1_tsr, input2_tsr, out_tsr);

    mli_mov_tensor_sync(out_tsr, &copy_config, out_slice.Sub());
    input1_slice.Next();
    input2_slice.Next();
    out_slice.Next();
  }
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}

void* AddInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus AddPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input1 =
      micro_context->AllocateTempInputTensor(node, kInputTensor1);
  TF_LITE_ENSURE(context, input1 != nullptr);
  TfLiteTensor* input2 =
      micro_context->AllocateTempInputTensor(node, kInputTensor2);
  TF_LITE_ENSURE(context, input2 != nullptr);
  TfLiteTensor* output = AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);

  TF_LITE_ENSURE_STATUS(
      CalculateOpData(context, params, input1, input2, output, data));

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus AddEval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus ret_val = kTfLiteOk;
  auto* params = reinterpret_cast<TfLiteAddParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kInputTensor1);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kInputTensor2);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  if (data->is_mli_applicable) {
    ret_val =
        EvalMLIAddInt8(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteFloat32) {
    ret_val = EvalAdd(context, node, params, data, input1, input2, output);
  } else if (output->type == kTfLiteInt8 || output->type == kTfLiteInt16) {
    ret_val =
        EvalAddQuantized(context, node, params, data, input1, input2, output);
  } else {
    MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(output->type),
                output->type);
    ret_val = kTfLiteError;
  }

  return ret_val;
}

TfLiteRegistration_V1 Register_ADD() {
  return tflite::micro::RegisterOp(AddInit, AddPrepare, AddEval);
}

}  // namespace tflite
