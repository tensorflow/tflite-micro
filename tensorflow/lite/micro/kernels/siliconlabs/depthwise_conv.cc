
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/micro_kernel_util.h"
#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"

#include "sl_mvp_ml_depthwise_conv2d.h"

namespace tflite {
namespace sl {
namespace depthwise_conv2d {

constexpr int kInputTensor  = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor   = 2;
constexpr int kOutputTensor = 0;

// Depthwise conv is quantized along dimension 3 of filter tensor.
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kDepthwiseConvQuantizedDimension = 3;

enum op_support { kMvp, kCmsisNN, kTFLMrefF32, kTFLMrefI8 };

struct OpData {
  op_support  supported;
  float       activation_min_f32;
  float       activation_max_f32;
  int         scratch_buffer_index;
  sli_mvp_ml_depthwise_conv2d_s8_params_t op_params;

  // CMSIS-NN per channel output multiplier and shift.
  int32_t     *per_channel_output_multiplier;
  int32_t     *per_channel_output_shift;
};

inline float16_t normalize_fp16(float f)
{
  return (float16_t)std::min(std::max(f, SLI_MVP_FP16_MIN), SLI_MVP_FP16_MAX);
}

inline PaddingType RuntimePaddingType(TfLitePadding padding)
{
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context,
    const TfLiteTensor* input,
    const TfLiteTensor* filter,
    TfLiteTensor* output,
    const TfLiteFusedActivation& activation,
    int32_t* output_activation_min, int32_t* output_activation_max,
    float16_t* per_channel_scalers, int num_channels, float accumulator_multipler)
{
  auto affine_quantization =
        reinterpret_cast<const TfLiteAffineQuantization*>(filter->quantization.params);

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;

  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float filter_scale = filter_scales[i];
    const float effective_output_scale = (input_scale * filter_scale) / output_scale;
    const float acc_output_scale = effective_output_scale * accumulator_multipler;
    per_channel_scalers[i] = normalize_fp16(acc_output_scale);
  }

  TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
          context, activation, output, output_activation_min,
          output_activation_max));

  return kTfLiteOk;
}

void *Init(TfLiteContext* context, const char* buffer, size_t length)
{
  (void)buffer;
  (void)length;
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node)
{
  int scratch_buffer_size = 0;

  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params = static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

  TfLiteTensor* output       = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* bias   = GetOptionalInputTensor(context, node, kBiasTensor);
  const TfLiteTensor* input  = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, filter != nullptr);

  data->op_params.batches         = input->dims->data[0];
  data->op_params.in_channels     = input->dims->data[3];
  data->op_params.input_height    = input->dims->data[1];
  data->op_params.input_width     = input->dims->data[2];
  data->op_params.out_channels    = filter->dims->data[kDepthwiseConvQuantizedDimension];
  data->op_params.output_height   = output->dims->data[1];
  data->op_params.output_width    = output->dims->data[2];
  data->op_params.filter_height   = filter->dims->data[1];
  data->op_params.filter_width    = filter->dims->data[2];
  data->op_params.input_offset    = -input->params.zero_point;
  data->op_params.output_offset   = output->params.zero_point;
  data->op_params.stride_height   = params->stride_height;
  data->op_params.stride_width    = params->stride_width;
  data->op_params.dilation_height = params->dilation_height_factor;
  data->op_params.dilation_width  = params->dilation_width_factor;
  data->op_params.padding         = params->padding == kTfLitePaddingSame;

  int dummy_height, dummy_width;
  const auto padding = ComputePaddingHeightWidth(
                         params->stride_height, params->stride_width,
                         params->dilation_height_factor, params->dilation_width_factor,
                         data->op_params.input_height, data->op_params.input_width,
                         data->op_params.filter_height, data->op_params.filter_width,
                         params->padding,
                         &dummy_height, &dummy_width);

  data->op_params.pad_height = padding.height;
  data->op_params.pad_width  = padding.width;

  const int num_channels = data->op_params.out_channels;

  if (input->type == kTfLiteInt8) {
    if (sli_mvp_ml_depthwise_conv2d_s8_is_supported(&data->op_params)) {
      data->supported = kMvp;

      float16_t *bias_data = static_cast<float16_t*>(context->AllocatePersistentBuffer(
                             context, num_channels * sizeof(float16_t)));
      if(bias != nullptr) {
        data->op_params.bias = bias_data;
        int32_t i32_bias;
        for(int i = 0; i < num_channels; i++) {
          i32_bias = bias->data.i32[i];
          bias_data[i] = float16_t(i32_bias * SLI_MVP_ACCUMULATOR_SCALER);
        }
      } else {
        data->op_params.bias = nullptr;
      }

      float16_t *scaler_data = static_cast<float16_t*>(context->AllocatePersistentBuffer(
                               context, num_channels * sizeof(float16_t)));
      data->op_params.output_scaler = scaler_data;
      TF_LITE_ENSURE_STATUS(PopulateConvolutionQuantizationParams(
        context, input, filter, output, params->activation,
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_min),
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_max),
        scaler_data, num_channels, SLI_MVP_ACCUMULATOR_MULTIPLIER));

    } else {
      data->per_channel_output_multiplier = static_cast<int32_t*>(context->AllocatePersistentBuffer(
                                            context, num_channels * sizeof(int32_t)));
      data->per_channel_output_shift = static_cast<int32_t*>(context->AllocatePersistentBuffer(
                                       context, num_channels * sizeof(int32_t)));

      int32_t dummy_output_multiplier;
      int dummy_output_shift;
      TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &dummy_output_multiplier, &dummy_output_shift,
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_min),
        reinterpret_cast<int32_t*>(&data->op_params.output_activation_max),
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        num_channels));

      if (data->op_params.dilation_height == 1 && data->op_params.dilation_width == 1) {
        data->supported = kCmsisNN;
        cmsis_nn_dw_conv_params       dw_conv_params;
        dw_conv_params.input_offset   = data->op_params.input_offset;
        dw_conv_params.output_offset  = data->op_params.output_offset;
        dw_conv_params.stride.h       = data->op_params.stride_height;
        dw_conv_params.stride.w       = data->op_params.stride_width;
        dw_conv_params.dilation.h     = 1;
        dw_conv_params.dilation.w     = 1;
        dw_conv_params.padding.h      = data->op_params.pad_height;
        dw_conv_params.padding.w      = data->op_params.pad_width;
        dw_conv_params.activation.min = data->op_params.output_activation_min;
        dw_conv_params.activation.max = data->op_params.output_activation_max;
        dw_conv_params.ch_mult        = data->op_params.out_channels / data->op_params.in_channels;

        cmsis_nn_dims input_dims;
        input_dims.n = data->op_params.batches;
        input_dims.h = data->op_params.input_height;
        input_dims.w = data->op_params.input_width;
        input_dims.c = data->op_params.in_channels;

        cmsis_nn_dims filter_dims;
        filter_dims.h = data->op_params.filter_height;
        filter_dims.w = data->op_params.filter_width;

        cmsis_nn_dims output_dims;
        output_dims.h = data->op_params.output_height;
        output_dims.w = data->op_params.output_width;
        output_dims.c = data->op_params.out_channels;

        scratch_buffer_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(
                              &dw_conv_params, &input_dims, &filter_dims, &output_dims);
      } else {
        data->supported = kTFLMrefI8;
      }
    }

  } else if (input->type == kTfLiteFloat32) {
    data->supported = kTFLMrefF32;
    CalculateActivationRange(params->activation,
                             &data->activation_min_f32,
                             &data->activation_max_f32);

  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                       TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  if(scratch_buffer_size > 0) {
    TF_LITE_ENSURE_STATUS(
      context->RequestScratchBufferInArena(
                 context, scratch_buffer_size, &data->scratch_buffer_index));
  } else {
    data->scratch_buffer_index = -1;
  }

  return kTfLiteOk;
}

TfLiteStatus eval_mvp_int8(TfLiteContext* context,
                           OpData* data,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           TfLiteEvalTensor* output)
{
  data->op_params.input  = tflite::micro::GetTensorData<int8_t>(input);
  data->op_params.output = tflite::micro::GetTensorData<int8_t>(output);
  data->op_params.filter = tflite::micro::GetTensorData<int8_t>(filter);

  TF_LITE_ENSURE_EQ(context, SL_STATUS_OK, sli_mvp_ml_depthwise_conv2d_s8(&data->op_params));

  return kTfLiteOk;
}

TfLiteStatus eval_cmsis_int8(TfLiteContext* context,
                             OpData* data,
                             const TfLiteEvalTensor* input,
                             const TfLiteEvalTensor* filter,
                             const TfLiteEvalTensor* bias,
                             TfLiteEvalTensor* output)
{
  cmsis_nn_dims input_dims;
  input_dims.n = data->op_params.batches;
  input_dims.h = data->op_params.input_height;
  input_dims.w = data->op_params.input_width;
  input_dims.c = data->op_params.in_channels;

  cmsis_nn_dims filter_dims;
  filter_dims.n = data->op_params.in_channels;
  filter_dims.h = data->op_params.filter_height;
  filter_dims.w = data->op_params.filter_width;
  filter_dims.c = data->op_params.out_channels;

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = data->op_params.out_channels;

  cmsis_nn_dims output_dims;
  output_dims.n = data->op_params.batches;
  output_dims.h = data->op_params.output_height;
  output_dims.w = data->op_params.output_width;
  output_dims.c = data->op_params.out_channels;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = data->per_channel_output_multiplier;
  quant_params.shift = data->per_channel_output_shift;

  cmsis_nn_dw_conv_params       dw_conv_params;
  dw_conv_params.input_offset   = data->op_params.input_offset;
  dw_conv_params.output_offset  = data->op_params.output_offset;
  dw_conv_params.stride.h       = data->op_params.stride_height;
  dw_conv_params.stride.w       = data->op_params.stride_width;
  dw_conv_params.dilation.h     = 1;
  dw_conv_params.dilation.w     = 1;
  dw_conv_params.padding.h      = data->op_params.pad_height;
  dw_conv_params.padding.w      = data->op_params.pad_width;
  dw_conv_params.activation.min = data->op_params.output_activation_min;
  dw_conv_params.activation.max = data->op_params.output_activation_max;
  dw_conv_params.ch_mult        = data->op_params.out_channels / data->op_params.in_channels;

  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data->scratch_buffer_index > -1) {
    ctx.buf = context->GetScratchBuffer(context, data->scratch_buffer_index);
  }
  TFLITE_DCHECK_EQ(ARM_MATH_SUCCESS,
                   arm_depthwise_conv_wrapper_s8(
                     &ctx, &dw_conv_params, &quant_params,
                     &input_dims,  tflite::micro::GetTensorData<int8_t>(input),
                     &filter_dims, tflite::micro::GetTensorData<int8_t>(filter),
                     &bias_dims,   bias == nullptr ? NULL : tflite::micro::GetTensorData<int32_t>(bias),
                     &output_dims, tflite::micro::GetTensorData<int8_t>(output)));

  return kTfLiteOk;
}

TfLiteStatus eval_tflm_int8(OpData* data,
                            const TfLiteEvalTensor* input,
                            const TfLiteEvalTensor* filter,
                            const TfLiteEvalTensor* bias,
                            TfLiteEvalTensor* output)
{
  DepthwiseParams dw_op_params;

  dw_op_params.input_offset             = data->op_params.input_offset;
  dw_op_params.output_offset            = data->op_params.output_offset;
  dw_op_params.stride_height            = data->op_params.stride_height;
  dw_op_params.stride_width             = data->op_params.stride_width;
  dw_op_params.dilation_height_factor   = data->op_params.dilation_height;
  dw_op_params.dilation_width_factor    = data->op_params.dilation_width;
  dw_op_params.padding_values.height    = data->op_params.pad_height;
  dw_op_params.padding_values.width     = data->op_params.pad_width;
  dw_op_params.quantized_activation_min = data->op_params.output_activation_min;
  dw_op_params.quantized_activation_max = data->op_params.output_activation_max;
  dw_op_params.depth_multiplier         = data->op_params.out_channels / data->op_params.in_channels;

  reference_integer_ops::DepthwiseConvPerChannel(
    dw_op_params,
    data->per_channel_output_multiplier,
    data->per_channel_output_shift,
    tflite::micro::GetTensorShape(input),
    tflite::micro::GetTensorData<int8_t>(input),
    tflite::micro::GetTensorShape(filter),
    tflite::micro::GetTensorData<int8_t>(filter),
    tflite::micro::GetTensorShape(bias),
    bias == nullptr ? nullptr : tflite::micro::GetTensorData<int32_t>(bias),
    tflite::micro::GetTensorShape(output),
    tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus eval_float(TfLiteDepthwiseConvParams* params,
                        const OpData* data,
                        const TfLiteEvalTensor* input,
                        const TfLiteEvalTensor* filter,
                        const TfLiteEvalTensor* bias,
                        TfLiteEvalTensor* output)
{
  DepthwiseParams dw_op_params;

  dw_op_params.padding_type           = RuntimePaddingType(params->padding);
  dw_op_params.padding_values.width   = data->op_params.pad_width;
  dw_op_params.padding_values.height  = data->op_params.pad_height;
  dw_op_params.stride_width           = data->op_params.stride_width;
  dw_op_params.stride_height          = data->op_params.stride_height;
  dw_op_params.dilation_width_factor  = data->op_params.dilation_width;
  dw_op_params.dilation_height_factor = data->op_params.dilation_height;
  dw_op_params.float_activation_min   = data->activation_min_f32;
  dw_op_params.float_activation_max   = data->activation_max_f32;
  dw_op_params.depth_multiplier       = data->op_params.out_channels / data->op_params.in_channels;

  reference_ops::DepthwiseConv(dw_op_params,
                               tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<float>(input),
                               tflite::micro::GetTensorShape(filter),
                               tflite::micro::GetTensorData<float>(filter),
                               tflite::micro::GetTensorShape(bias),
                               bias == nullptr ? nullptr : tflite::micro::GetTensorData<float>(bias),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node)
{
  TfLiteStatus status = kTfLiteError;

  TFLITE_DCHECK(node->user_data    != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params = reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const auto input  = tflite::micro::GetEvalInput(context, node, kInputTensor);
  const auto filter = tflite::micro::GetEvalInput(context, node, kFilterTensor);
  const auto bias   = NumInputs(node) == 3
                      ? tflite::micro::GetEvalInput(context, node, kBiasTensor)
                      : nullptr;
  auto output       = tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (data->supported == kMvp) {
    status = eval_mvp_int8(context, data, input, filter, output);

  } else if (data->supported == kCmsisNN) {
    status = eval_cmsis_int8(context, data, input, filter, bias, output);

  } else if (data->supported == kTFLMrefI8) {
    status = eval_tflm_int8(data, input, filter, bias, output);

  } else if (data->supported == kTFLMrefF32) {
    status = eval_float(params, data, input, filter, bias, output);
  }

  return status;
}

}  // namespace depthwise_conv2d
}  // namespace sl

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
  return {/*init=*/sl::depthwise_conv2d::Init,
          /*free=*/nullptr,
          /*prepare=*/sl::depthwise_conv2d::Prepare,
          /*invoke=*/sl::depthwise_conv2d::Invoke,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
