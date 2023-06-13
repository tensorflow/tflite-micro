#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "cmsis/CMSIS/NN/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_kernel_util.h"
#include "sl_mvp_ml_fully_connected.h"

namespace tflite {
namespace sl {
namespace fully_connected {

struct OpData {
  int32_t output_multiplier;
  int output_shift;
  sli_mvp_ml_fully_connected_s8_params_t op_params;
  float16_t *bias_fp16;
  bool use_mvp;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

// TODO(b/169801227): This global struct is needed for the linker to drop unused
// code (for example, by using Register_FULLY_CONNECTED_INT8 instead of
// Register_FULLY_CONNECTED).
TfLiteRegistration fully_connected_registration;

sli_shape_t dims2shape(const TfLiteIntArray *dim)
{
  TFLITE_DCHECK(dim->size <= 4);

  sli_shape_t shape = {0};
  for (int i = 0; i < dim->size; i++) {
    shape.dim[i] = dim->data[i];
  }
  return shape;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  TfLiteFullyConnectedParams* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  const TfLiteTensor* input  = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weight = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias   = GetInput(context, node, kBiasTensor);
  TfLiteTensor*       output = GetOutput(context, node, kOutputTensor);
  int32_t             output_min;
  int32_t             output_max;
  float16_t           *bias_data = nullptr;
  int                 bias_len = 0;

  TF_LITE_ENSURE(context, input  != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);

  if (!(input->type == kTfLiteFloat32 || input->type == kTfLiteInt8)) {
    // Unsupported datatype used by model
    return kTfLiteError;
  }

  if (bias) {
    RuntimeShape bias_shape = GetTensorShape(bias);
    bias_len = bias_shape.FlatSize();
  }

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
    context, params->activation, output, &output_min, &output_max));

    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weight, bias, output, &real_multiplier));

    data->op_params.input = GetTensorData<int8_t>(input);
    data->op_params.input_shape = dims2shape(input->dims);
    data->op_params.input_offset = -input->params.zero_point;
    data->op_params.weight = GetTensorData<int8_t>(weight);
    data->op_params.weight_shape = dims2shape(weight->dims);
    data->op_params.weight_offset = -weight->params.zero_point;
    data->op_params.bias = nullptr;
    data->op_params.bias_length = bias_len;
    data->op_params.output = GetTensorData<int8_t>(output);
    data->op_params.output_shape = dims2shape(output->dims);
    data->op_params.output_offset = output->params.zero_point;
    data->op_params.output_multiplier = sli_mvp_ml_fully_connected_output_multiplier(real_multiplier);
    data->op_params.activation_min = static_cast<int8_t>(output_min);
    data->op_params.activation_max = static_cast<int8_t>(output_max);

    data->use_mvp = sli_mvp_ml_fully_connected_s8_is_supported(&data->op_params);

    if (data->use_mvp && bias) {
      // Convert int32_t to float16_t as the MVP does not support loading int32 values.
      const int32_t *bias_src = GetTensorData<int32_t>(bias);
      bias_data = static_cast<float16_t *>(context->AllocatePersistentBuffer(context, bias_len * sizeof(float16_t)));
      if (bias_data == nullptr) {
        return kTfLiteError;
      }
      sl_status_t status = sli_mvp_ml_fully_connected_bias_convert(bias_src, bias_data, bias_len);
      if (status != SL_STATUS_OK) {
        return kTfLiteError;
      }
      data->op_params.bias = bias_data;
    }

    if (!data->use_mvp) {
      // In this case we have to convert the output scale factor to a
      // value in the TensorFlow fixed point format (Q.31 + shift)
      int exponent;
      QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
      data->output_shift = -exponent;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8_MVP(TfLiteContext* context, TfLiteNode* node,
                               const OpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  sli_mvp_ml_fully_connected_s8_params_t *params = const_cast<sli_mvp_ml_fully_connected_s8_params_t*>(&data.op_params);
  params->input  = tflite::micro::GetTensorData<int8_t>(input);
  params->output = tflite::micro::GetTensorData<int8_t>(output);
  
  sl_status_t result = sli_mvp_ml_fully_connected_s8(params);
  if (result == SL_STATUS_OK) {
    return kTfLiteOk;
  } else {
    return kTfLiteError;
  }
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               const OpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  if (data.use_mvp && input->type == kTfLiteInt8) {
    return EvalQuantizedInt8_MVP(context, node, data, input, filter, bias, output);
  }

  // The 'if' condition can be removed when null handling of bias is added to
  // arm_fully_connected_s8
  if (nullptr != tflite::micro::GetTensorData<int32_t>(bias)) {
    const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
    const int batches = output_shape.Dims(0);
    const int output_depth = output_shape.Dims(1);
    const RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    const int filter_dim_count = filter_shape.DimensionsCount();
    const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
    const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);

    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = data.op_params.input_offset;
    fc_params.output_offset = data.op_params.output_offset;
    fc_params.filter_offset = data.op_params.weight_offset;
    fc_params.activation.min = data.op_params.activation_min;
    fc_params.activation.max = data.op_params.activation_max;

    cmsis_nn_per_tensor_quant_params quant_params;
    quant_params.multiplier = data.output_multiplier;
    // TODO(b/138810107): Figure out whether output shift should be inverted
    quant_params.shift = -data.output_shift;

    cmsis_nn_dims input_dims;
    input_dims.n = batches;
    input_dims.h = 1;
    input_dims.w = 1;
    input_dims.c = accum_depth;

    cmsis_nn_dims filter_dims;
    filter_dims.n = accum_depth;
    filter_dims.h = 1;
    filter_dims.w = 1;
    filter_dims.c = output_depth;

    cmsis_nn_dims bias_dims;
    bias_dims.n = 1;
    bias_dims.h = 1;
    bias_dims.w = 1;
    bias_dims.c = output_depth;

    cmsis_nn_dims output_dims;
    output_dims.n = batches;
    output_dims.h = 1;
    output_dims.w = 1;
    output_dims.c = output_depth;

    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    ctx.size = 0;

    TF_LITE_ENSURE_EQ(
        context,
        arm_fully_connected_s8(
            &ctx, &fc_params, &quant_params, &input_dims,
            tflite::micro::GetTensorData<int8_t>(input), &filter_dims,
            tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
            tflite::micro::GetTensorData<int32_t>(bias), &output_dims,
            tflite::micro::GetTensorData<int8_t>(output)),
        ARM_MATH_SUCCESS);
  } else {
    tflite::FullyConnectedParams op_params;
    op_params.input_offset = data.op_params.input_offset;
    op_params.weights_offset = data.op_params.weight_offset;
    op_params.output_offset = data.op_params.output_offset;
    op_params.output_multiplier = data.output_multiplier;
    // TODO(b/138810107): Figure out whether output shift should be inverted
    op_params.output_shift = -data.output_shift;
    op_params.quantized_activation_min = data.op_params.activation_min;
    op_params.quantized_activation_max = data.op_params.activation_max;

    reference_integer_ops::FullyConnected(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFusedActivation activation,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
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
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params->activation, input, filter, bias,
                       output);
    case kTfLiteInt8:
      return EvalQuantizedInt8(context, node, data, input, filter, bias,
                               output);

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

// Note that the current function names are not ideal at all (this EvalInt8
// function internally calls EvalQuantizedInt8, and there is similar name
// aliasing in the Eval function too). We will be attempting to have a more
// descriptive naming convention but holding off on that for now, since the
// renaming might be coupled with reducing code duplication and some additional
// refactoring.
TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  if (input->type != kTfLiteInt8) {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }

  return EvalQuantizedInt8(context, node, data, input, filter, bias, output);
}

}  // namespace fully_connected
}  // namespace sl

TfLiteRegistration Register_FULLY_CONNECTED() {
  return {/*init*/sl::fully_connected::Init,
          /*free*/nullptr,
          /*prepare*/sl::fully_connected::Prepare,
          /*invoke*/sl::fully_connected::Eval,
          /*profiling_string*/nullptr,
          /*builtin_code*/0,
          /*custom_name*/nullptr,
          /*version*/0};
}

TfLiteRegistration Register_FULLY_CONNECTED_INT8() {
  return {/*init*/sl::fully_connected::Init,
          /*free*/nullptr,
          /*prepare*/sl::fully_connected::Prepare,
          /*invoke*/sl::fully_connected::EvalInt8,
          /*profiling_string*/nullptr,
          /*builtin_code*/0,
          /*custom_name*/nullptr,
          /*version*/0};
}

}  // namespace tflite
