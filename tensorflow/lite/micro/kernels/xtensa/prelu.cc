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

#include "tensorflow/lite/kernels/internal/reference/prelu.h"

#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/prelu.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_prelu.h"
#endif

namespace tflite {

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
  return context->AllocatePersistentBuffer(context, sizeof(XtensaPreluData));
#else
  return context->AllocatePersistentBuffer(context, sizeof(PreluParams));
#endif  
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const PreluParams& params =
      *(static_cast<const PreluParams*>(node->user_data));

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* alpha = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteFloat32: {
      BroadcastPrelu4DSlowFloat(tflite::micro::GetTensorShape(input),
                                tflite::micro::GetTensorData<float>(input),
                                tflite::micro::GetTensorShape(alpha),
                                tflite::micro::GetTensorData<float>(alpha),
                                tflite::micro::GetTensorShape(output),
                                tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
#if defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
    case kTfLiteInt8: {
      const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape& alpha_shape = tflite::micro::GetTensorShape(alpha);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const RuntimeShape extended_alpha_shape = RuntimeShape::ExtendedShape(output_shape.DimensionsCount(), alpha_shape);
      
      XtensaPreluData* data = static_cast<XtensaPreluData*>(node->user_data);
      const int8_t *alpha_data = tflite::micro::GetTensorData<int8_t>(alpha);
      int8_t *alpha_broadcasted;
      if(!(alpha_shape == output_shape) && input_shape == output_shape){
        alpha_broadcasted = (int8_t *)context->GetScratchBuffer(context, data->scratch_tensor_index);
        xa_nn_broadcast_8_8(alpha_broadcasted, output_shape.DimsData(), tflite::micro::GetTensorData<int8_t>(alpha), extended_alpha_shape.DimsData(), extended_alpha_shape.DimensionsCount());
        alpha_data = alpha_broadcasted;
      } 

      if(input_shape == output_shape)
      {
        int err = 0;
        const int flat_size = MatchingFlatSize(input_shape, output_shape);
        err = xa_nn_vec_prelu_asym8s_asym8s(
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorData<int8_t>(input),
          alpha_data,
          params.input_offset,
          params.alpha_offset,
          params.output_multiplier_2,
          params.output_shift_2,
          params.output_multiplier_1,
          params.output_shift_1,
          params.output_offset,
	  flat_size);

    TF_LITE_ENSURE(context, err == 0);
      }
      else
      {
        reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      }
      return kTfLiteOk;
    } break;
#else
    case kTfLiteInt8: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
      return kTfLiteOk;
    } break;
#endif // defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ)
    case kTfLiteInt16: {
      reference_ops::BroadcastPrelu4DSlow(
          params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int16_t>(input),
          tflite::micro::GetTensorShape(alpha),
          tflite::micro::GetTensorData<int8_t>(alpha),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int16_t>(output));
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Only float32 and uint8_t are supported currently, got %d.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TFLMRegistration Register_PRELU() {
  return tflite::micro::RegisterOp(PreluInit, PreluPrepare, PreluEval);
}

}  // namespace tflite
