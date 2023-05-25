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

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
  // Cached tensor zero point values for quantized operations.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

  // The result of checking if MLI optimized version of tensors can be used.
  bool is_mli_applicable;

  // Tensors in MLI format.
  mutable ops::micro::MliTensorInterface mli_in;
  mutable ops::micro::MliTensorInterface mli_weights;
  mutable ops::micro::MliTensorInterface mli_bias;
  mutable ops::micro::MliTensorInterface mli_out;

#ifdef MLI_2_0
  mli_fully_connected_cfg* cfg;
#endif
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

bool IsMliApplicable(TfLiteContext* context, const TfLiteTensor* input,
                     const TfLiteTensor* filter, const TfLiteTensor* bias,
                     const TfLiteFullyConnectedParams* params,
                     int32_t output_activation_min,
                     int32_t output_activation_max) {
  // MLI optimized version only supports int8_t datatype and no fused Relu and
  // symmetric per-tensor quantization of weights (not per-axis)
  bool ret_val =
      (filter->type == kTfLiteInt8) && (input->type == kTfLiteInt8) &&
      (bias->type == kTfLiteInt32) &&
#ifndef MLI_2_0
      (params->activation == kTfLiteActNone ||
       (output_activation_min == -128 && output_activation_max == 127)) &&
#endif
      (filter->params.zero_point == 0);
  return ret_val;
}

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             const TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  if (data_type != kTfLiteFloat32 && !data->is_mli_applicable) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
#endif
  return status;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kWeightsTensor);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = AllocateTempOutputTensor(node, kOutputTensor);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  TfLiteStatus status = CalculateOpData(context, params, input->type, input,
                                        filter, bias, output, data);

  data->is_mli_applicable =
      IsMliApplicable(context, input, filter, bias, params,
                      data->output_activation_min, data->output_activation_max);

  if (input->type == kTfLiteInt8 && data->is_mli_applicable) {
    data->mli_in = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
    data->mli_weights = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
    data->mli_bias = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));
    data->mli_out = ops::micro::MliTensorInterface(static_cast<mli_tensor*>(
        context->AllocatePersistentBuffer(context, sizeof(mli_tensor))));

    ops::micro::ConvertToMliTensor(input, &data->mli_in);
    ops::micro::ConvertToMliTensor(filter, &data->mli_weights);
    ops::micro::ConvertToMliTensor(bias, &data->mli_bias);
#ifdef MLI_2_0
    ops::micro::AdjustBiasTensor(&data->mli_bias, &data->mli_in,
                                 &data->mli_weights);
#endif
    ops::micro::ConvertToMliTensor(output, &data->mli_out);

#ifdef MLI_2_0
    if (data->output_activation_min == -128 &&
        data->output_activation_max == 127) {
      data->cfg->relu.type = MLI_RELU_NONE;
    } else if (params->activation == kTfLiteActRelu) {
      data->cfg->relu.type = MLI_RELU_GEN;
    } else if (params->activation == kTfLiteActRelu6) {
      data->cfg->relu.type = MLI_RELU_6;
    } else if (params->activation == kTfLiteActReluN1To1) {
      data->cfg->relu.type = MLI_RELU_1;
    } else {
      data->cfg->relu.type = MLI_RELU_NONE;
    }
#endif

    /* The input tensor can have more than 2 dimensions. for the compute this
   doesn't make any difference because all the inputs or a batch entry will
   be used anyway. because the MLI kernel doesn't recognize the multiple
   dimensions, the tensor shape is casted to a {batchnum, inputsize} shape. */
    data->mli_in.Shape()[0] = data->mli_out.Shape()[0];
#if defined(MLI_2_0) && !defined(MLI_2_0_KRNL_TEST)
    data->mli_in.Shape()[1] = data->mli_weights.Shape()[0];
#else
    data->mli_in.Shape()[1] = data->mli_weights.Shape()[1];
#endif
    data->mli_in.Shape()[2] = 0;
    data->mli_in.Shape()[3] = 0;
    data->mli_in.MemStride()[0] = data->mli_in.Shape()[1];
    data->mli_in.MemStride()[1] = 0;
    *data->mli_in.Rank() = 2;
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(bias);
  micro_context->DeallocateTempTfLiteTensor(output);
  return status;
}

TfLiteStatus EvalMliQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                                  const TfLiteFullyConnectedParams* params,
                                  const OpData& data,
                                  const TfLiteEvalTensor* input,
                                  const TfLiteEvalTensor* filter,
                                  const TfLiteEvalTensor* bias,
                                  TfLiteEvalTensor* output) {
  ops::micro::MliTensorAttachBuffer<int8_t>(input, &data.mli_in);
  ops::micro::MliTensorAttachBuffer<int8_t>(filter, &data.mli_weights);
  ops::micro::MliTensorAttachBuffer<int32_t>(bias, &data.mli_bias);
  ops::micro::MliTensorAttachBuffer<int8_t>(output, &data.mli_out);

  // Tensors for data in fast (local) memory and config to copy data from
  // external to local memory
  mli_tensor weights_local = *data.mli_weights.MliTensor();
  mli_tensor bias_local = *data.mli_bias.MliTensor();
  mli_tensor in_local = *data.mli_in.MliTensor();
  mli_tensor out_local = *data.mli_out.MliTensor();

  ops::micro::MliTensorInterface weights_local_interface(&weights_local);
  ops::micro::MliTensorInterface bias_local_interface(&bias_local);
  ops::micro::MliTensorInterface in_local_interface(&in_local);
  ops::micro::MliTensorInterface out_local_interface(&out_local);

  mli_mov_cfg_t copy_config;
  mli_mov_cfg_for_copy(&copy_config);
#if defined(MLI_2_0) && !defined(MLI_2_0_KRNL_TEST)
  const int weight_out_dimension = 1;
#else
  const int weight_out_dimension = 0;
#endif
  // bias has only 1 dimension
  const int bias_out_ch_dimension = 0;
  const int out_tensor_dimension = 1;
  const int input_size_dimension = 1;
  int slice_size = data.mli_weights.Shape()[weight_out_dimension];

  /* allocate the local buffers, and compute the slice size */
  TF_LITE_ENSURE_STATUS(
      ops::micro::get_arc_scratch_buffer_for_fully_connect_tensors(
          context, &in_local_interface, &weights_local_interface,
          &bias_local_interface, &out_local_interface));
  TF_LITE_ENSURE_STATUS(ops::micro::arc_scratch_buffer_calc_slice_size_weights(
      &weights_local_interface, &bias_local_interface, weight_out_dimension,
      &slice_size));

  int max_out_slice_size = *out_local_interface.DataCapacity() /
                           mli_hlp_tensor_element_size(&out_local);

  if (slice_size > max_out_slice_size) slice_size = max_out_slice_size;

  /* is_local indicates that the tensor is already in local memory,
     so in that case the original tensor can be used,
     and there is no need to copy it to the local tensor*/
  const bool in_is_local =
      in_local_interface.Data<int8_t>() == data.mli_in.Data<int8_t>();
  const bool out_is_local =
      out_local_interface.Data<int8_t>() == data.mli_out.Data<int8_t>();
  const bool b_is_local =
      bias_local_interface.Data<int32_t>() == data.mli_bias.Data<int32_t>();
#ifndef MLI_2_0_KRNL_TEST
  const bool w_is_local =
      weights_local_interface.Data<int8_t>() == data.mli_weights.Data<int8_t>();
#endif

#if defined(MLI_2_0) && !defined(MLI_2_0_KRNL_TEST)
  ops::micro::TensorSlicer w_slice(data.mli_weights.MliTensor(),
                                   weight_out_dimension, slice_size, 0, 0, 0,
                                   true);
#else
  ops::micro::TensorSlicer w_slice(data.mli_weights.MliTensor(),
                                   weight_out_dimension, slice_size);
#endif
  ops::micro::TensorSlicer b_slice(data.mli_bias.MliTensor(),
                                   bias_out_ch_dimension, slice_size);
  ops::micro::TensorSlicer out_ch_slice(data.mli_out.MliTensor(),
                                        out_tensor_dimension, slice_size, 0, 0,
                                        0, true);

#ifdef MLI_2_0_KRNL_TEST
  mli_tensor* w_ptr = &weights_local;
#else
  mli_tensor* w_ptr = w_is_local ? w_slice.Sub() : &weights_local;
#endif
  mli_tensor* b_ptr = b_is_local ? b_slice.Sub() : &bias_local;

  void* input_buffer_ptr = NULL;

  while (!w_slice.Done()) {
#if defined(MLI_2_0) && !defined(MLI_2_0_KRNL_TEST)
    w_ptr->el_params.sa.scale.mem.pi16 = NULL;
    b_ptr->el_params.sa.scale.mem.pi16 = NULL;
#endif

#ifndef MLI_2_0_KRNL_TEST
    mli_mov_tensor_sync(w_slice.Sub(), &copy_config, w_ptr);
#endif
    mli_mov_tensor_sync(b_slice.Sub(), &copy_config, b_ptr);

    // Slice the input over the batches (one at a time with the size of a
    // complete input)
    ops::micro::TensorSlicer in_slice(
        data.mli_in.MliTensor(), input_size_dimension,
        data.mli_in.Shape()[input_size_dimension]);

    /* output tensor is already sliced in the output size dimension.
    out_ch_slice.Sub() is the tensor for the amount of output size of this
    iteration of the weight slice loop. This tensor needs to be further
    sliced over the batch */
    ops::micro::TensorSlicer out_slice(out_ch_slice.Sub(), out_tensor_dimension,
                                       slice_size);

    /* setup the pointers to the local or remote tensor to make the code
     * inside the loop easier. */
    mli_tensor* in_ptr = in_is_local ? in_slice.Sub() : &in_local;
    mli_tensor* out_ptr = out_is_local ? out_slice.Sub() : &out_local;

#ifdef MLI_2_0_KRNL_TEST
    /* Permute weights tensor to the HWCN layout */
    // Assertion here to prevent usage non-contiguous buffer memory.
    if (data.mli_out.Shape()[out_tensor_dimension] !=
        out_slice.Sub()->shape[0]) {
      MicroPrintf("Slicing is not supported with real-time permutation.");
      return kTfLiteError;
    }
    mli_permute_cfg permute_cfg = {{1, 0, 2, 3}};
    ops::micro::permute_weights(data.mli_weights.MliTensor(), &permute_cfg,
                                w_ptr, &out_ptr->data);
#endif

    while (!out_slice.Done()) {
      if (!out_is_local) {
        ops::micro::PrepareLocalTensor(out_slice.Sub(), &out_local);
        ops::micro::PrepareLocalTensor(in_slice.Sub(), &in_local);
      }
      // if same input copy as previous iteration, skip the copy of input
#ifdef MLI_2_0
      if (in_slice.Sub()->data.mem.pi8 != input_buffer_ptr) {
        mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
        input_buffer_ptr = in_slice.Sub()->data.mem.pi8;
      }
      mli_fully_connected_cfg cfg;
      cfg.relu.type = MLI_RELU_NONE;
      mli_krn_fully_connected_sa8_sa8_sa32(in_ptr, w_ptr, b_ptr, &cfg, out_ptr);
#else
      if (in_slice.Sub()->data != input_buffer_ptr) {
        mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
        input_buffer_ptr = in_slice.Sub()->data;
      }
      mli_krn_fully_connected_sa8_sa8_sa32(in_ptr, w_ptr, b_ptr, out_ptr);
#endif

      mli_mov_tensor_sync(out_ptr, &copy_config, out_slice.Sub());

      in_slice.Next();
      out_slice.Next();
    }
    w_slice.Next();
    b_slice.Next();
    out_ch_slice.Next();
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           const OpData& data, const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier = data.output_multiplier;
  op_params.output_shift = -data.output_shift;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

  reference_integer_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
#else
  MicroPrintf("Node configuration is not supported by ARC MLI Library.");
  return kTfLiteError;
#endif
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFusedActivation activation,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
#else
  MicroPrintf("Type %s (%d) is not supported by ARC MLI Library.",
              TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kBiasTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params->activation, input, filter, bias,
                       output);
    case kTfLiteInt8:
      if (data.is_mli_applicable) {
        return EvalMliQuantizedInt8(context, node, params, data, input, filter,
                                    bias, output);
      } else {
        return EvalQuantized(context, node, data, input, filter, bias, output);
      }

    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TFLMRegistration Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
