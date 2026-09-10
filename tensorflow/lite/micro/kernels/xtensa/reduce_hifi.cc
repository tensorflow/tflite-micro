/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_reduce.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

const int kMaxNumberOfAxisHifi = 5;
const int kMaxNumberOfReducedAxisHifi = 4;

extern TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node,
                           int32_t* multiplier, int* shift);
extern void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params);

TfLiteStatus PrepareMaxHifi(TfLiteContext* context, TfLiteNode* node,
                              OpDataReduce* op_data) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node, &op_data->multiplier,
                                           &op_data->shift));

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);

  op_data->input_scale = input->params.scale;
  op_data->output_scale = output->params.scale;
  op_data->num_output_elements = NumElements(output);

  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->scratch_accumulator_idx);
  context->RequestScratchBufferInArena(
      context, sizeof(int) * static_cast<int>(ElementCount(*axis->dims)),
      &op_data->scratch_resolved_axis_idx);
#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  XtensaReduceOpData* xt_data =
          reinterpret_cast<XtensaReduceOpData*>(node->user_data);
  if((input->dims->size <= 4) && (input->type == kTfLiteInt8))
  {
    reduce_ops_t reduce_type = REDUCE_MAX;
    const TfLiteEvalTensor* eval_axis = tflite::micro::GetEvalInput(context, node, 1);
    const int *axis_data_ptr  = tflite::micro::GetTensorData<int>(eval_axis);
    int num_axis = static_cast<int>(ElementCount(*eval_axis->dims));
    int required_scratch;
    required_scratch = xa_nn_reduce_getsize_nhwc(-4,
                                                input->dims->data,
                                                input->dims->size,
                                                axis_data_ptr,
                                                num_axis,
                                                reduce_type);
    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
                        "reduce_max_4D_asym8: xa_nn_reduce_getsize_nhwc failed");
      return kTfLiteError;
    }

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, required_scratch,
        &(xt_data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);
  }
#endif
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

TfLiteStatus PrepareMeanOrSumHifi(TfLiteContext* context, TfLiteNode* node,
                                    OpDataReduce* op_data) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TfLiteTensor* axis = micro_context->AllocateTempInputTensor(node, 1);
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const double real_multiplier = static_cast<double>(input->params.scale) /
                                   static_cast<double>(output->params.scale);
    QuantizeMultiplier(real_multiplier, &op_data->multiplier, &op_data->shift);
  }

  int output_size = NumElements(output);
  op_data->num_axis = NumElements(axis);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    context->RequestScratchBufferInArena(context, output_size * sizeof(int32_t),
                                         &op_data->scratch_accumulator_idx);
    op_data->input_zp = input->params.zero_point;
    op_data->input_scale = input->params.scale;
    op_data->output_zp = output->params.zero_point;
    op_data->output_scale = output->params.scale;
  }
  context->RequestScratchBufferInArena(context, sizeof(int) * input->dims->size,
                                       &op_data->scratch_input_iter_idx);
  context->RequestScratchBufferInArena(context, sizeof(int) * op_data->num_axis,
                                       &op_data->scratch_resolved_axis_idx);
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  XtensaReduceOpData* xt_data =
          reinterpret_cast<XtensaReduceOpData*>(node->user_data);
  if((input->dims->size <= 4) && (input->type == kTfLiteInt8 || input->type == kTfLiteInt16))
  {
    reduce_ops_t reduce_type = REDUCE_MEAN;
    const TfLiteEvalTensor* eval_axis = tflite::micro::GetEvalInput(context, node, 1);
    const int *axis_data_ptr  = tflite::micro::GetTensorData<int>(eval_axis);
    int num_axis = static_cast<int>(ElementCount(*eval_axis->dims));
    int required_scratch;
    const TfLiteEvalTensor* eval_input = tflite::micro::GetEvalInput(context, node, 0);
    int resolved_axis[kMaxNumberOfReducedAxisHifi];
    int num_resolved_axis = 0;
    if (!reference_ops::ResolveAxis(eval_input->dims->size, tflite::micro::GetTensorData<int>(eval_axis), num_axis, resolved_axis, &num_resolved_axis)) {
      TF_LITE_ENSURE(context, false);
    }
    int num_elm_in_axis = 1;
    int axis_itr;
    
    for(axis_itr=0; axis_itr < num_resolved_axis; axis_itr++)
    {
      num_elm_in_axis *= eval_input->dims->data[resolved_axis[axis_itr]];
    }

    int shift = 63 - CountLeadingZeros(static_cast<uint64_t>(num_elm_in_axis));
    shift = std::min(std::min(shift, 32), 31 + op_data->shift);
    xt_data->updated_multiplier = (num_elm_in_axis > 1) ? (WORD32)(((long long int)(op_data->multiplier) << shift) / num_elm_in_axis) : op_data->multiplier;
    xt_data->updated_shift = op_data->shift - shift;
    
    int inp_precision = -4; /* ASYM8S */ 
    if(input->type == kTfLiteInt16)
     inp_precision = -7; /* ASYM16S */ 
    
    required_scratch = xa_nn_reduce_getsize_nhwc(inp_precision,
                                                input->dims->data,
                                                input->dims->size,
                                                axis_data_ptr,
                                                num_axis,
                                                reduce_type);
    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
                        "reduce_mean_4D_asym8: xa_nn_reduce_getsize_nhwc failed");
      return kTfLiteError;
    }

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, required_scratch,
        &(xt_data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);
    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(output);
    micro_context->DeallocateTempTfLiteTensor(axis);
    return kTfLiteOk;
  }
#endif
  TF_LITE_ENSURE_OK(
      context,
      PrepareSimple(context, node, &(op_data->multiplier), &(op_data->shift)));
  // TODO(b/144955155): Support uint8_t(b/144955155) and int8_t(b/144955018)
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(axis);
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus QuantizedMeanOrSum(TfLiteContext* context, TfLiteNode* node,
                                int* temp_index, int* resolved_axis,
                                int32_t* temp_sum, OpDataReduce* op_data,
                                bool compute_sum) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  bool result = reference_ops::QuantizedMeanOrSumExtraArgs<T, int32_t>(
      tflite::micro::GetTensorData<T>(input), op_data->input_zp,
      op_data->input_scale, &input->dims->data[0], input->dims->size,
      tflite::micro::GetTensorData<T>(output), op_data->output_scale,
      op_data->multiplier, op_data->shift, op_data->output_zp,
      &output->dims->data[0], output->dims->size,
      tflite::micro::GetTensorData<int>(axis), op_data->num_axis,
      params->keep_dims, temp_index, resolved_axis, temp_sum, compute_sum);
  TF_LITE_ENSURE(context, result);

  return kTfLiteOk;
}

template <typename T, typename U>
TfLiteStatus Mean(TfLiteContext* context, TfLiteNode* node,
                  OpDataReduce* op_data, int* temp_index, int* resolved_axis,
                  U* temp_sum) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  reference_ops::Mean<T, U>(
      tflite::micro::GetTensorData<T>(input), &input->dims->data[0],
      input->dims->size, tflite::micro::GetTensorData<T>(output),
      &output->dims->data[0], output->dims->size,
      tflite::micro::GetTensorData<int>(axis), op_data->num_axis,
      params->keep_dims, temp_index, resolved_axis, temp_sum);

  return kTfLiteOk;
}

template <typename integer_type>
TfLiteStatus EvalIntegerMean(TfLiteContext* context, TfLiteNode* node,
                             int num_axis, OpDataReduce* op_data,
                             int* temp_index, int* resolved_axis) {
  int32_t* temp_sum = static_cast<int32_t*>(
      context->GetScratchBuffer(context, op_data->scratch_accumulator_idx));

  if (op_data->input_zp == op_data->output_zp &&
      op_data->input_scale == op_data->output_scale) {
    Mean<integer_type, int32_t>(context, node, op_data, temp_index,
                                resolved_axis, temp_sum);
  } else {
    QuantizedMeanOrSum<integer_type>(context, node, temp_index, resolved_axis,
                                     temp_sum, op_data, /*compute_sum=*/false);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMeanHifi(TfLiteContext* context, TfLiteNode* node,
                            OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TfLiteReducerParams* params =
      reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);

  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int temp_index[kMaxNumberOfAxisHifi];
  int resolved_axis[kMaxNumberOfReducedAxisHifi];

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::MeanParams op_params;
      ResolveAxis(tflite::micro::GetTensorData<int>(axis), num_axis,
                  &op_params);

      // Special case mean implementation exists for 4D mean across axes 1
      // and 2.
      bool special_case_4d_axes_1_and_2 =
          input->dims->size == 4 && op_params.axis_count == 2 &&
          ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
           (op_params.axis[0] == 2 && op_params.axis[1] == 1));

      // Defer to specialized implementation for 4D Mean across axes 1 & 2.
      if (params->keep_dims && special_case_4d_axes_1_and_2) {
        reference_ops::Mean(op_params, tflite::micro::GetTensorShape(input),
                            tflite::micro::GetTensorData<float>(input),
                            tflite::micro::GetTensorShape(output),
                            tflite::micro::GetTensorData<float>(output));
      } else {
        TF_LITE_ENSURE(
            context,
            reference_ops::Mean(
                tflite::micro::GetTensorData<float>(input), input->dims->data,
                input->dims->size, tflite::micro::GetTensorData<float>(output),
                output->dims->data, output->dims->size,
                tflite::micro::GetTensorData<int>(axis), num_axis,
                params->keep_dims, temp_index, resolved_axis,
                tflite::micro::GetTensorData<float>(output)));
      }
    } break;
    case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
      XtensaReduceOpData* xt_data =
              reinterpret_cast<XtensaReduceOpData*>(node->user_data);
      const int8_t *input_data_ptr  = tflite::micro::GetTensorData<int8_t>(input);
      int8_t *output_data_ptr  = tflite::micro::GetTensorData<int8_t>(output);
      void *p_scratch;
      int err = 0;

      if(input->dims->size <= 4)
      {
        // Resolve axis.
        int num_resolved_axis = 0;
        if (!reference_ops::ResolveAxis(input->dims->size, tflite::micro::GetTensorData<int>(axis), num_axis, resolved_axis, &num_resolved_axis)) {
          TF_LITE_ENSURE(context, false);
        }
        p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, xt_data->scratch_tensor_index));

        int output_size = output->dims->size;
        int output_dims_data[1] = {1};
        int* out_dims_ptr = output->dims->data;
        if(output_size == 0){
          output_size = 1;
          out_dims_ptr = output_dims_data;
        }
        err = xa_nn_reduce_mean_4D_asym8s_asym8s(output_data_ptr,
                                                 out_dims_ptr,
                                                 input_data_ptr,
                                                 input->dims->data,
                                                 resolved_axis,
                                                 output_size,
                                                 input->dims->size,
                                                 num_resolved_axis,
                                                 op_data->input_zp,
                                                 xt_data->updated_multiplier,
                                                 xt_data->updated_shift,
                                                 op_data->output_zp,
                                                 p_scratch);
        TF_LITE_ENSURE(context, err == 0);
      }
      else
      {
        TF_LITE_ENSURE_OK(
            context, EvalIntegerMean<int8_t>(context, node, num_axis, op_data,
                                             temp_index, resolved_axis));
      }
#else
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int8_t>(context, node, num_axis, op_data,
                                           temp_index, resolved_axis));
#endif
    } break;
    case kTfLiteInt16: {
#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
      XtensaReduceOpData* xt_data =
              reinterpret_cast<XtensaReduceOpData*>(node->user_data);
      const int16_t *input_data_ptr  = tflite::micro::GetTensorData<int16_t>(input);
      int16_t *output_data_ptr  = tflite::micro::GetTensorData<int16_t>(output);
      void *p_scratch;
      int err = 0;

      if(input->dims->size <= 4)
      {
        // Resolve axis.
        int num_resolved_axis = 0;
        if (!reference_ops::ResolveAxis(input->dims->size, tflite::micro::GetTensorData<int>(axis), num_axis, resolved_axis, &num_resolved_axis)) {
          TF_LITE_ENSURE(context, false);
        }
        p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, xt_data->scratch_tensor_index));

        err = xa_nn_reduce_mean_4D_asym16s_asym16s(output_data_ptr,
                                                 output->dims->data,
                                                 input_data_ptr,
                                                 input->dims->data,
                                                 resolved_axis,
                                                 output->dims->size,
                                                 input->dims->size,
                                                 num_resolved_axis,
                                                 op_data->input_zp,
                                                 xt_data->updated_multiplier,
                                                 xt_data->updated_shift,
                                                 op_data->output_zp,
                                                 p_scratch);
        TF_LITE_ENSURE(context, err == 0);
      }
      else
      {
        TF_LITE_ENSURE_OK(
            context, EvalIntegerMean<int16_t>(context, node, num_axis, op_data,
                                             temp_index, resolved_axis));
      }
#else
      TF_LITE_ENSURE_OK(
          context, EvalIntegerMean<int16_t>(context, node, num_axis, op_data,
                                            temp_index, resolved_axis));
#endif
    } break;
    default:
      TF_LITE_ENSURE_MSG(context, false,
                         "Currently, only float32, int8 or int16 input type "
                         "is supported.");
  }
  return kTfLiteOk;
}

TfLiteStatus EvalMaxHifi(TfLiteContext* context, TfLiteNode* node,
                           OpDataReduce* op_data) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* axis = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TfLiteReducerParams* params =
      static_cast<TfLiteReducerParams*>(node->builtin_data);

  // Interpret an axis tensor with null dimensions as a scalar
  int num_axis = static_cast<int>(ElementCount(*axis->dims));
  int* temp_buffer = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_accumulator_idx));
  int* resolved_axis = static_cast<int*>(
      context->GetScratchBuffer(context, op_data->scratch_resolved_axis_idx));
  switch (input->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<float>(
              tflite::micro::GetTensorData<float>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<float>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              std::numeric_limits<float>::lowest(),
              [](const float current, const float in) -> float {
                return (in > current) ? in : current;
              }));
      break;
    case kTfLiteInt8: {
#if (defined(HIFI_IQ) || defined(HIFI5) || defined(HIFI4))
      XtensaReduceOpData* xt_data =
              reinterpret_cast<XtensaReduceOpData*>(node->user_data);
      const int8_t *input_data_ptr  = tflite::micro::GetTensorData<int8_t>(input);
      int8_t *output_data_ptr  = tflite::micro::GetTensorData<int8_t>(output);
      void *p_scratch;
      int err = 0;

      if(input->dims->size <= 4)
      {
        // Resolve axis.
        int num_resolved_axis = 0;
        if (!reference_ops::ResolveAxis(input->dims->size, tflite::micro::GetTensorData<int>(axis), num_axis, resolved_axis, &num_resolved_axis)) {
          TF_LITE_ENSURE(context, false);
        }
        p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, xt_data->scratch_tensor_index));

        err = xa_nn_reduce_max_4D_asym8s_asym8s(output_data_ptr,
                                                output->dims->data,
                                                input_data_ptr,
                                                input->dims->data,
                                                resolved_axis,
                                                output->dims->size,
                                                input->dims->size,
                                                num_resolved_axis,
                                                p_scratch);
        TF_LITE_ENSURE(context, err == 0);
      }
      else
      {
        TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                          static_cast<double>(op_data->output_scale));
        TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
        TF_LITE_ENSURE(
            context,
            reference_ops::ReduceGeneric<int8_t>(
                tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
                input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
                output->dims->data, output->dims->size,
                tflite::micro::GetTensorData<int>(axis), num_axis,
                params->keep_dims, temp_buffer, resolved_axis,
                std::numeric_limits<int8_t>::lowest(),
                [](const int8_t current, const int8_t in) -> int8_t {
                  return (in > current) ? in : current;
                }));
      }
#else
      TF_LITE_ENSURE_EQ(context, static_cast<double>(op_data->input_scale),
                        static_cast<double>(op_data->output_scale));
      TF_LITE_ENSURE_EQ(context, op_data->input_zp, op_data->output_zp);
      TF_LITE_ENSURE(
          context,
          reference_ops::ReduceGeneric<int8_t>(
              tflite::micro::GetTensorData<int8_t>(input), input->dims->data,
              input->dims->size, tflite::micro::GetTensorData<int8_t>(output),
              output->dims->data, output->dims->size,
              tflite::micro::GetTensorData<int>(axis), num_axis,
              params->keep_dims, temp_buffer, resolved_axis,
              std::numeric_limits<int8_t>::lowest(),
              [](const int8_t current, const int8_t in) -> int8_t {
                return (in > current) ? in : current;
              }));
#endif
    } break;
    default:
      MicroPrintf("Only float32, int8, and int16 types are supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace tflite
