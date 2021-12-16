#if defined(VISIONP6)

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_depthwise_conv.h"

namespace tflite {

TfLiteStatus DepthwiseConvPrepareXtensa(TfLiteContext* context,
                                        TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  XtensaDepthwiseConvOpData* data =
      reinterpret_cast<XtensaDepthwiseConvOpData*>(node->user_data);
  const auto& params =
      *(reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data));

  TfLiteTensor* output = GetOutput(context, node, kDepthwiseConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input =
      GetInput(context, node, kDepthwiseConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter =
      GetInput(context, node, kDepthwiseConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  const TfLiteTensor* bias = GetInput(context, node, kDepthwiseConvBiasTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  data->per_channel_output_shift_int8 = static_cast<int8_t*>(
      context->AllocatePersistentBuffer(context, num_channels));

  for (int i = 0; i < num_channels; i++) {
    data->per_channel_output_shift_int8[i] = static_cast<int8_t>(
        -1 * data->reference_op_data.per_channel_output_shift[i]);
  }

  uint32_t context_size = 0;
  uint32_t status = xiDepthwiseConvGetMemReqd_Context(&context_size);
  if (!status && context_size) {
    void* context_data =
        context->AllocatePersistentBuffer(context, context_size);
    if (context_data == nullptr) {
      return kTfLiteError;
    }
    data->p_context = (uint8_t*)context_data;
    data->context_size = context_size;
  }

  const uint32_t input_height = SizeOfDimension(input, 1);
  const uint32_t input_width = SizeOfDimension(input, 2);
  const uint32_t input_depth = SizeOfDimension(input, 3);

  const uint32_t output_height = SizeOfDimension(output, 1);
  const uint32_t output_width = SizeOfDimension(output, 2);
  const uint32_t output_depth = SizeOfDimension(output, 3);

  const uint32_t filter_height = SizeOfDimension(filter, 1);
  const uint32_t filter_width = SizeOfDimension(filter, 2);

  status = xiDepthwiseConvSetContext(
      data->p_context, data->context_size, input_depth, input_width,
      input_height, output_depth, output_width, output_height, filter_width,
      filter_height, params.stride_width, input->params.zero_point,
      filter->params.zero_point, output->params.zero_point,
      data->reference_op_data.output_multiplier,
      data->reference_op_data.output_shift,
      data->reference_op_data.output_activation_min,
      data->reference_op_data.output_activation_max);
  if (status) {
    return kTfLiteError;
  }

  uint32_t coefficent_size = 0;
  status = xiDepthwiseConvGetMemReqd_Coeff(data->p_context, data->context_size,
                                           &coefficent_size);
  if (status || coefficent_size == 0) {
    return kTfLiteError;
  }

  void* coeff_data =
      context->AllocatePersistentBuffer(context, coefficent_size);
  if (coeff_data == nullptr) {
    return kTfLiteError;
  }
  data->reorder_coefficient_bias = reinterpret_cast<int8_t*>(coeff_data);
  data->reorder_coefficient_bias_size = coefficent_size;

  status = xiDepthwiseConvDoCoeffReorder(
      data->p_context, data->context_size,
      reinterpret_cast<uint8_t*>(data->reorder_coefficient_bias),
      data->reorder_coefficient_bias_size,
      const_cast<uint8_t*>(GetTensorData<uint8_t>(filter)),
      const_cast<int32_t*>(GetTensorData<int32_t>(bias)));
  if (status) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus DepthwiseConvEvalXtensa(TfLiteContext* context, TfLiteNode* node,
                                     const TfLiteDepthwiseConvParams& params,
                                     const XtensaDepthwiseConvOpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output) {
  uint32_t input_size = input->dims->data[0] * input->dims->data[1] *
                        input->dims->data[2] * input->dims->data[3];
  uint32_t output_size = output->dims->data[0] * output->dims->data[1] *
                         output->dims->data[2] * output->dims->data[3];
  uint32_t num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  xiDepthwiseConv(
      data.p_context, data.context_size,
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input)),
      input_size, tflite::micro::GetTensorData<int8_t>(output), output_size,
      data.reorder_coefficient_bias, data.reorder_coefficient_bias_size,
      data.reference_op_data.per_channel_output_multiplier,
      data.per_channel_output_shift_int8, num_channels,
      data.reference_op_data.padding.width,
      data.reference_op_data.padding.height);
  return kTfLiteOk;
}
}  // namespace tflite
#endif  // defined(VISIONP6)
