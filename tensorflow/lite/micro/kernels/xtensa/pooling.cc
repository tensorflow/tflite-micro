/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
  int32_t activation_min;
  int32_t activation_max;
  float activation_min_f32;
  float activation_max_f32;
  int scratch_tensor_index;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

TfLiteStatus AverageEvalFloat(TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
#if HIFI_VFPU
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int kernel_height = params->filter_height;
  const int kernel_width = params->filter_width;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const float* inp_data_ptr;
  float* out_data_ptr;
  int inp_data_format = 0, out_data_format = 0, out_length;
  void* p_scratch;
  int err;

  p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  inp_data_ptr = tflite::micro::GetTensorData<float>(input);
  out_data_ptr = tflite::micro::GetTensorData<float>(output);

  for (int batch = 0; batch < batches; ++batch) {
    err = xa_nn_avgpool_f32(
        &out_data_ptr[output_height * output_width * depth * batch],
        &inp_data_ptr[output_height * output_width * depth * batch],
        input_height, input_width, depth, kernel_height, kernel_width,
        stride_width, stride_height, pad_width, pad_height, output_height,
        output_width, inp_data_format, out_data_format, p_scratch);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  }

  out_length = batches * output_height * output_width * depth;
  err = xa_nn_vec_activation_min_max_f32_f32(
      out_data_ptr, out_data_ptr, data->activation_min_f32,
      data->activation_max_f32, out_length);

  TF_LITE_MICRO_EXPECT_EQ(0, err);
#else
  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<float>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<float>(output));
#endif // HIFI_VFPU
  return kTfLiteOk;
}

#if defined(HIFI5)
TfLiteStatus AverageEvalQuantized(TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const TfLitePoolParams* params,
                                  const OpData* data,
                                  const TfLiteEvalTensor* input,
                                  TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);

  if (input->type == kTfLiteUInt8) {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const uint8_t* inp_data_ptr;
    uint8_t* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    void* p_scratch;
    int err;

    p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data->scratch_tensor_index));

    inp_data_ptr = tflite::micro::GetTensorData<uint8_t>(input);
    out_data_ptr = tflite::micro::GetTensorData<uint8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_avgpool_asym8(
          &out_data_ptr[output_height * output_width * depth * batch],
          &inp_data_ptr[output_height * output_width * depth * batch],
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }

    out_length = batches * output_height * output_width * depth;
    err = xa_nn_vec_activation_min_max_asym8_asym8(out_data_ptr, out_data_ptr,
                                                   data->activation_min,
                                                   data->activation_max, out_length);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  } else {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const int8_t* inp_data_ptr;
    int8_t* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    void* p_scratch;
    int err;

    p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data->scratch_tensor_index));

    inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
    out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_avgpool_8(
          &out_data_ptr[output_height * output_width * depth * batch],
          const_cast <int8_t *>(&inp_data_ptr[output_height * output_width * depth * batch]),
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }

    out_length = batches * output_height * output_width * depth;
    err = xa_nn_vec_activation_min_max_8_8(out_data_ptr, out_data_ptr,
                                                   data->activation_min,
                                                   data->activation_max, out_length);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  }
  return kTfLiteOk;
}
#else
void AverageEvalQuantized(TfLiteContext* context, const TfLiteNode* node,
                          const TfLitePoolParams* params, const OpData* data,
                          const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  if (input->type == kTfLiteUInt8) {
    reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<uint8_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<uint8_t>(output));
  } else {
    reference_integer_ops::AveragePool(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
}
#endif // defined(HIFI5)

TfLiteStatus MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, const OpData* data,
                  const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
#if HIFI_VFPU
  const int stride_height = params->stride_height;
  const int stride_width = params->stride_width;
  const int pad_width = data->padding.width;
  const int pad_height = data->padding.height;
  const int kernel_height = params->filter_height;
  const int kernel_width = params->filter_width;

  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  const float* inp_data_ptr;
  float* out_data_ptr;
  int inp_data_format = 0, out_data_format = 0, out_length;
  void* p_scratch;
  int err;

  p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, data->scratch_tensor_index));

  inp_data_ptr = tflite::micro::GetTensorData<float>(input);
  out_data_ptr = tflite::micro::GetTensorData<float>(output);

  for (int batch = 0; batch < batches; ++batch) {
    err = xa_nn_maxpool_f32(
        &out_data_ptr[output_height * output_width * depth * batch],
        &inp_data_ptr[output_height * output_width * depth * batch],
        input_height, input_width, depth, kernel_height, kernel_width,
        stride_width, stride_height, pad_width, pad_height, output_height,
        output_width, inp_data_format, out_data_format, p_scratch);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  }

  out_length = batches * output_height * output_width * depth;
  err = xa_nn_vec_activation_min_max_f32_f32(out_data_ptr, out_data_ptr,
                                             data->activation_min_f32,
                                             data->activation_max_f32, out_length);

  TF_LITE_MICRO_EXPECT_EQ(0, err);
#else
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<float>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<float>(output));
#endif // HIFI_VFPU
  return kTfLiteOk;
}

#if defined(HIFI5)
TfLiteStatus MaxEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLitePoolParams* params, const OpData* data,
                              const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  if (input->type == kTfLiteUInt8) {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const uint8_t* inp_data_ptr;
    uint8_t* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    void* p_scratch;
    int err;

    p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data->scratch_tensor_index));

    inp_data_ptr = tflite::micro::GetTensorData<uint8_t>(input);
    out_data_ptr = tflite::micro::GetTensorData<uint8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_maxpool_asym8(
          &out_data_ptr[output_height * output_width * depth * batch],
          &inp_data_ptr[output_height * output_width * depth * batch],
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }

    out_length = batches * output_height * output_width * depth;
    err = xa_nn_vec_activation_min_max_asym8_asym8(
        out_data_ptr, out_data_ptr, data->activation_min, data->activation_max,
        out_length);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  } else {
    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;

    const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
    const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
    const int batches = MatchingDim(input_shape, 0, output_shape, 0);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    const int8_t* inp_data_ptr;
    int8_t* out_data_ptr;
    int inp_data_format = 0, out_data_format = 0, out_length;
    void* p_scratch;
    int err;

    p_scratch = static_cast<void*>(
        context->GetScratchBuffer(context, data->scratch_tensor_index));

    inp_data_ptr = tflite::micro::GetTensorData<int8_t>(input);
    out_data_ptr = tflite::micro::GetTensorData<int8_t>(output);

    for (int batch = 0; batch < batches; ++batch) {
      err = xa_nn_maxpool_8(
          &out_data_ptr[output_height * output_width * depth * batch],
          const_cast <int8_t *>(&inp_data_ptr[output_height * output_width * depth * batch]),
          input_height, input_width, depth, kernel_height, kernel_width,
          stride_width, stride_height, pad_width, pad_height, output_height,
          output_width, inp_data_format, out_data_format, p_scratch);

      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }

    out_length = batches * output_height * output_width * depth;
    err = xa_nn_vec_activation_min_max_8_8(
        out_data_ptr, out_data_ptr, data->activation_min, data->activation_max,
        out_length);

    TF_LITE_MICRO_EXPECT_EQ(0, err);
  }
  return kTfLiteOk;
}
#else
void MaxEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                      TfLitePoolParams* params, const OpData* data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  if (input->type == kTfLiteUInt8) {
    reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<uint8_t>(input),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<uint8_t>(output));
  } else {
    reference_integer_ops::MaxPool(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
}
#endif // defined(HIFI5)
}  // namespace

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      AverageEvalQuantized(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      MaxEvalQuantized(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

#if defined(HIFI5)
// Using separate Prepare functions for Averagepool and maxpool
// because of different calls to get_size functions.
TfLiteStatus AveragePrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, data));

  if (input->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation, &data->activation_min_f32,
                             &data->activation_max_f32);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &data->activation_min,
                                      &data->activation_max);
  }

  if ((input->type == kTfLiteInt8) ||
      (input->type == kTfLiteUInt8) ||
      (input->type == kTfLiteFloat32)) {
    int input_precision, output_precision;

    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;
    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& output_shape = GetTensorShape(output);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    if (input->type == kTfLiteInt8) {
      output_precision = input_precision = PREC_8;
    } else if (input->type == kTfLiteUInt8) {
      output_precision = input_precision = PREC_ASYM8;
    } else {
      output_precision = input_precision = PREC_F32;
    }

    int required_scratch = xa_nn_avgpool_getsize(
        depth, input_precision, output_precision, input_height, input_width,
        kernel_height, kernel_width,
        stride_width,   // x_stride,
        stride_height,  // y_stride,
        pad_width,      // x_padding,
        pad_height,     // y_padding,
        output_height, output_width, 0 /*NHWC input */, 0 /* NHWC output */);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
          "Averagepool: xa_nn_avgpool_getsize failed");
      return kTfLiteError;
    }

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, required_scratch,
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);

  }
  return kTfLiteOk;
}

TfLiteStatus MaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, data));

  if (input->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation, &data->activation_min_f32,
                             &data->activation_max_f32);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &data->activation_min,
                                      &data->activation_max);
  }

  if ((input->type == kTfLiteInt8) ||
      (input->type == kTfLiteUInt8) ||
      (input->type == kTfLiteFloat32)) {
    int input_precision, output_precision;

    const int stride_height = params->stride_height;
    const int stride_width = params->stride_width;
    const int pad_width = data->padding.width;
    const int pad_height = data->padding.height;
    const int kernel_height = params->filter_height;
    const int kernel_width = params->filter_width;
    const RuntimeShape& input_shape = GetTensorShape(input);
    const RuntimeShape& output_shape = GetTensorShape(output);
    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    if (input->type == kTfLiteInt8) {
      output_precision = input_precision = PREC_8;
    } else if (input->type == kTfLiteUInt8) {
      output_precision = input_precision = PREC_ASYM8;
    } else {
      output_precision = input_precision = PREC_F32;
    }

    int required_scratch = xa_nn_maxpool_getsize(
        depth, input_precision, output_precision, input_height, input_width,
        kernel_height, kernel_width,
        stride_width,   // x_stride,
        stride_height,  // y_stride,
        pad_width,      // x_padding,
        pad_height,     // y_padding,
        output_height, output_width, 0 /* NHWC inpput */, 0 /* NHWC output */);

    if (required_scratch <= 0) {
      TF_LITE_KERNEL_LOG(context,
          "Maxpool: xa_nn_maxpool_getsize failed");
      return kTfLiteError;
    }

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, required_scratch,
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);

  }
  return kTfLiteOk;
}

#else

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, data));

  if (input->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation, &data->activation_min_f32,
                             &data->activation_max_f32);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &data->activation_min,
                                      &data->activation_max);
  }

  return kTfLiteOk;
}
#endif // defined(HIFI5)

}  // namespace pooling

TfLiteRegistration Register_AVERAGE_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
#if defined(HIFI5)
          /*prepare=*/pooling::AveragePrepare,
#else
          /*prepare=*/pooling::Prepare,
#endif
          /*invoke=*/pooling::AverageEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MAX_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
#if defined(HIFI5)
          /*prepare=*/pooling::MaxPrepare,
#else
          /*prepare=*/pooling::Prepare,
#endif
          /*invoke=*/pooling::MaxEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
