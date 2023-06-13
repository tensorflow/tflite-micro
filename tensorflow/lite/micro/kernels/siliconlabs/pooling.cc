
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/micro_kernel_util.h"

#include "sl_mvp_ml_pooling.h"

namespace tflite {
namespace sl {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum op_support { kMvp, kCmsisNN, kTFLMrefF32};

struct OpData {
  float activation_min_f32;
  float activation_max_f32;
  sli_mvp_ml_pooling_s8_params_t op_params;
  op_support supported;
  int buffer_idx;
};

}  // namespace


void* Init(TfLiteContext* context, const char* buffer, size_t length)
{
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}


TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node)
{
  OpData* data = static_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  const TfLiteTensor* input  = GetInput(context, node, kInputTensor);
  TfLiteTensor*       output = GetOutput(context, node, kOutputTensor);

  data->op_params.padding       = params->padding == kTfLitePaddingSame;
  data->op_params.stride_height = params->stride_height;
  data->op_params.stride_width  = params->stride_width;
  data->op_params.filter_height = params->filter_height;
  data->op_params.filter_width  = params->filter_width;
  data->op_params.batches       = MatchingDim(GetTensorShape(input),  0,
                                              GetTensorShape(output), 0);
  data->op_params.channels      = MatchingDim(GetTensorShape(input),  3,
                                              GetTensorShape(output), 3);
  data->op_params.input_height  = SizeOfDimension(input,  1);
  data->op_params.input_width   = SizeOfDimension(input,  2);
  data->op_params.output_height = SizeOfDimension(output, 1);
  data->op_params.output_width  = SizeOfDimension(output, 2);

  int out_height, out_width;
  auto padding = ComputePaddingHeightWidth(
                   params->stride_height, params->stride_width,
                   1, 1,  // dilation rate height/width.
                   data->op_params.input_height, data->op_params.input_width,
                   params->filter_height, params->filter_width,
                   params->padding,
                   &out_height, &out_width);
  TFLITE_DCHECK_EQ(out_height, data->op_params.output_height);
  TFLITE_DCHECK_EQ(out_width, data->op_params.output_width);
  data->op_params.pad_height = padding.height;
  data->op_params.pad_width  = padding.width;

  if (input->type == kTfLiteFloat32) {
    data->supported = kTFLMrefF32;
    CalculateActivationRange(params->activation,
                             &data->activation_min_f32,
                             &data->activation_max_f32);
  } else {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      reinterpret_cast<int32_t*>(&data->op_params.output_activation_min),
                                      reinterpret_cast<int32_t*>(&data->op_params.output_activation_max));
    if (input->type != kTfLiteInt8) {
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus AveragePrepare(TfLiteContext* context, TfLiteNode* node)
{
  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const TfLiteTensor* input  = GetInput(context, node, kInputTensor);
  TfLiteTensor*       output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);

  TfLiteStatus status = Prepare(context, node);

  if (status == kTfLiteOk) {
    if (input->type == kTfLiteInt8) {
      data->supported = sli_mvp_ml_average_pooling_s8_is_supported(&data->op_params)
                        ? kMvp : kCmsisNN;
      if (data->supported == kCmsisNN) {
        const int32_t buffer_size = arm_avgpool_s8_get_buffer_size(
                                      data->op_params.output_width,
                                      data->op_params.channels);

        if (buffer_size > 0) {
          TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
                                  context, buffer_size, &data->buffer_idx));
        } else {
          data->buffer_idx = -1;
        }
      }
    }
  }
  return status;
}

TfLiteStatus MaxPrepare(TfLiteContext* context, TfLiteNode* node)
{
  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const TfLiteTensor* input  = GetInput(context, node, kInputTensor);
  TfLiteTensor*       output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);

  TfLiteStatus status = Prepare(context, node);

  if (status == kTfLiteOk) {
    if (input->type == kTfLiteInt8) {
      data->supported = sli_mvp_ml_max_pooling_s8_is_supported(&data->op_params)
                        ? kMvp : kCmsisNN;
    }
  }

  return status;
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node)
{
  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const TfLiteEvalTensor* input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor*       output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  data->op_params.input  = tflite::micro::GetTensorData<int8_t>(input);
  data->op_params.output = tflite::micro::GetTensorData<int8_t>(output);

  if (data->supported == kMvp) {
    // Use MVP accelerated kernel.
    TF_LITE_ENSURE_EQ(context,
                      SL_STATUS_OK,
                      sli_mvp_ml_average_pooling_s8(&data->op_params));

  } else if (data->supported == kCmsisNN) {
    // Use CMSIS-NN optimized kernel.
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = data->op_params.input_height;
    input_dims.w = data->op_params.input_width;
    input_dims.c = data->op_params.channels;

    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = data->op_params.output_height;
    output_dims.w = data->op_params.output_width;
    output_dims.c = data->op_params.channels;

    cmsis_nn_pool_params pool_params;
    pool_params.stride.h = data->op_params.stride_height;
    pool_params.stride.w = data->op_params.stride_width;
    pool_params.padding.h = data->op_params.pad_height;
    pool_params.padding.w = data->op_params.pad_width;
    pool_params.activation.min = data->op_params.output_activation_min;
    pool_params.activation.max = data->op_params.output_activation_max;

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = data->op_params.filter_height;
    filter_dims.w = data->op_params.filter_width;
    filter_dims.c = 1;

    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    ctx.size = 0;
    if (data->buffer_idx > -1) {
      ctx.buf = context->GetScratchBuffer(context, data->buffer_idx);
    }

    TFLITE_DCHECK_EQ(
        arm_avgpool_s8(&ctx, &pool_params, &input_dims,
                       data->op_params.input, &filter_dims,
                       &output_dims,
                       data->op_params.output),
        ARM_MATH_SUCCESS);
  } else if (data->supported == kTFLMrefF32) {
    // Use TFLM reference kernel.
    tflite::PoolParams op_params;
    op_params.stride_height         = data->op_params.stride_height;
    op_params.stride_width          = data->op_params.stride_width;
    op_params.filter_height         = data->op_params.filter_height;
    op_params.filter_width          = data->op_params.filter_width;
    op_params.padding_values.height = data->op_params.pad_height;
    op_params.padding_values.width  = data->op_params.pad_width;
    op_params.float_activation_min  = data->activation_min_f32;
    op_params.float_activation_max  = data->activation_max_f32;
    reference_ops::AveragePool(op_params,
                               tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));

  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node)
{
  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const TfLiteEvalTensor* input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor*       output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  data->op_params.input  = tflite::micro::GetTensorData<int8_t>(input);
  data->op_params.output = tflite::micro::GetTensorData<int8_t>(output);

  if (data->supported == kMvp) {
    // Use MVP accelerated kernel.
    TF_LITE_ENSURE_EQ(context,
                      SL_STATUS_OK,
                      sli_mvp_ml_max_pooling_s8(&data->op_params));

  } else if (data->supported == kCmsisNN) {
    // Use CMSIS-NN optimized kernel.
    cmsis_nn_dims input_dims;
    input_dims.n = 1;
    input_dims.h = data->op_params.input_height;
    input_dims.w = data->op_params.input_width;
    input_dims.c = data->op_params.channels;

    cmsis_nn_dims output_dims;
    output_dims.n = 1;
    output_dims.h = data->op_params.output_height;
    output_dims.w = data->op_params.output_width;
    output_dims.c = data->op_params.channels;

    cmsis_nn_pool_params pool_params;
    pool_params.stride.h = data->op_params.stride_height;
    pool_params.stride.w = data->op_params.stride_width;
    pool_params.padding.h = data->op_params.pad_height;
    pool_params.padding.w = data->op_params.pad_width;
    pool_params.activation.min = data->op_params.output_activation_min;
    pool_params.activation.max = data->op_params.output_activation_max;

    cmsis_nn_dims filter_dims;
    filter_dims.n = 1;
    filter_dims.h = data->op_params.filter_height;
    filter_dims.w = data->op_params.filter_width;
    filter_dims.c = 1;

    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    ctx.size = 0;

    TFLITE_DCHECK_EQ(
        arm_max_pool_s8(&ctx, &pool_params, &input_dims,
                        data->op_params.input, &filter_dims,
                        &output_dims,
                        data->op_params.output),
        ARM_MATH_SUCCESS);
  } else if (data->supported == kTFLMrefF32) {
    // Use TFLM reference kernel.
    tflite::PoolParams op_params;
    op_params.stride_height         = data->op_params.stride_height;
    op_params.stride_width          = data->op_params.stride_width;
    op_params.filter_height         = data->op_params.filter_height;
    op_params.filter_width          = data->op_params.filter_width;
    op_params.padding_values.height = data->op_params.pad_height;
    op_params.padding_values.width  = data->op_params.pad_width;
    op_params.float_activation_min  = data->activation_min_f32;
    op_params.float_activation_max  = data->activation_max_f32;
    reference_ops::MaxPool(op_params,
                           tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<float>(input),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<float>(output));

  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace pooling
}  // namespace sl

TfLiteRegistration Register_MAX_POOL_2D() {
  static TfLiteRegistration max_pool_registration = {
    /*init=*/sl::pooling::Init,
    /*free=*/nullptr,
    /*prepare=*/sl::pooling::MaxPrepare,
    /*invoke=*/sl::pooling::MaxEval,
    /*profiling_string=*/nullptr,
    /*builtin_code=*/0,
    /*custom_name=*/nullptr,
    /*version=*/0
  };

  return max_pool_registration;
}

// Just to keep all_ops_resolver() happy during development ...
TfLiteRegistration Register_AVERAGE_POOL_2D() {
  static TfLiteRegistration avg_pool_registration = {
    /*init=*/sl::pooling::Init,
    /*free=*/nullptr,
    /*prepare=*/sl::pooling::AveragePrepare,
    /*invoke=*/sl::pooling::AverageEval,
    /*profiling_string=*/nullptr,
    /*builtin_code=*/0,
    /*custom_name=*/nullptr,
    /*version=*/0
  };

  return avg_pool_registration;
}

}  // namespace tflite
