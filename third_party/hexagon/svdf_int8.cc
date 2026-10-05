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

/* Copyright 2020 The Qualcomm Innovation Center, Inc. All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Qualcomm Innovation Center, Inc. nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================*/

#include <math.h>

#include "tensorflow/lite/micro/c/builtin_op_data.h"
#include "tensorflow/lite/micro/c/common.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/internal/common.h"
#include "tensorflow/lite/micro/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/op_macros.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "third_party/hexagon/hexagon_svdf.h"
#include "third_party/hexagon/hexagon_tflm_translation_svdf.h"

namespace tflite {
namespace micro {

namespace {

void HexagonSvdfOptimizationEvaluation(TfLiteContext* context, TfLiteNode* node,
                                       HexagonOpDataSvdf* data) {
  const TfLiteEvalTensor* input = GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      GetEvalInput(context, node, kSvdfWeightsTimeTensor);

  const int n_input = input->dims->data[1];
  const int n_filter = weights_feature->dims->data[0];
  const int n_memory = weights_time->dims->data[1];

  if (input->type == kTfLiteInt8 && (n_filter % 4 == 0) && (n_input % 8 == 0) &&
      (n_memory % 4 == 0)) {
    data->optimizable = 1;
  } else {
    data->optimizable = 0;
  }
}

TfLiteStatus HexagonSvdfOptimizedPrepare(TfLiteContext* context,
                                         TfLiteNode* node,
                                         HexagonOpDataSvdf* data) {
  const auto* params = static_cast<const TfLiteSVDFParams*>(node->builtin_data);
  const TfLiteEvalTensor* input = GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      GetEvalInput(context, node, kSvdfWeightsFeatureTensor);

  const int n_rank = params->rank;
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];
  const int n_filter = weights_feature->dims->data[0];
  const int n_unit = n_filter / n_rank;

  data->converted_bias = static_cast<int32_t*>(
      context->AllocatePersistentBuffer(context, n_filter * sizeof(int32_t)));
  TF_LITE_ENSURE(context, data->converted_bias != nullptr);

  int8_t* weights_feature_data =
      const_cast<int8_t*>(GetTensorData<int8_t>(weights_feature));
  HexagonGenerateBias(data->converted_bias, weights_feature_data,
                      /*bias=*/nullptr,
                      data->reference_op_data.input_zero_point + 128, n_filter,
                      1, n_input);
  HexagonInterleaveWeightInplace(weights_feature_data, n_filter, n_input, 1);

  TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                 context, n_batch * n_input * sizeof(uint8_t),
                                 &data->input_u8_scratch_index));
  TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                 context, n_batch * n_filter * sizeof(int32_t),
                                 &data->feature_s32_scratch_index));
  TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                 context, n_batch * n_filter * sizeof(int32_t),
                                 &data->time_s32_scratch_index));
  TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                 context, n_batch * n_unit * sizeof(int32_t),
                                 &data->output_s32_scratch_index));
  return kTfLiteOk;
}

void HexagonSvdfOptimizedEvalInt8(
    TfLiteContext* context, const TfLiteEvalTensor* input_tensor,
    const TfLiteEvalTensor* weights_feature_tensor,
    const TfLiteEvalTensor* weights_time_tensor,
    const TfLiteEvalTensor* bias_tensor, const TfLiteSVDFParams* params,
    TfLiteEvalTensor* activation_state_tensor, TfLiteEvalTensor* output_tensor,
    const HexagonOpDataSvdf& data) {
  const int n_rank = params->rank;
  const int n_batch = input_tensor->dims->data[0];
  const int n_input = input_tensor->dims->data[1];
  const int n_filter = weights_feature_tensor->dims->data[0];
  const int n_unit = n_filter / n_rank;
  const int n_memory = weights_time_tensor->dims->data[1];

  uint8_t* input_u8_scratch = static_cast<uint8_t*>(
      context->GetScratchBuffer(context, data.input_u8_scratch_index));
  int32_t* time_s32_scratch = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.time_s32_scratch_index));
  int32_t* feature_s32_scratch = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.feature_s32_scratch_index));
  int32_t* output_s32_scratch = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.output_s32_scratch_index));

  int16_t* const state_ptr = GetTensorData<int16_t>(activation_state_tensor);
  memmove(state_ptr, state_ptr + 1,
          (n_batch * n_filter * n_memory - 1) * sizeof(int16_t));

  const int8_t* input_data = GetTensorData<int8_t>(input_tensor);
  for (int i = 0; i < n_batch * n_input; ++i) {
    input_u8_scratch[i] = static_cast<uint8_t>(input_data[i] + 128);
  }

  const int8_t* weights_feature_data =
      GetTensorData<int8_t>(weights_feature_tensor);
  const int32_t state_max = std::numeric_limits<int16_t>::max();
  const int32_t state_min = std::numeric_limits<int16_t>::min();
  for (int b = 0; b < n_batch; ++b) {
    gemm_s32_s8xu8_Nany_Mmod4_Kmod8(
        weights_feature_data, input_u8_scratch + b * n_input,
        feature_s32_scratch + b * n_filter, n_filter, 1, n_input);

    int16_t* result_in_batch =
        state_ptr + b * n_memory * n_filter + (n_memory - 1);
    for (int r = 0; r < n_filter; ++r) {
      int32_t dot_prod =
          feature_s32_scratch[b * n_filter + r] + data.converted_bias[r];
      dot_prod = MultiplyByQuantizedMultiplier(
          dot_prod, data.reference_op_data.effective_scale_1_a,
          data.reference_op_data.effective_scale_1_b);
      dot_prod = std::min(std::max(state_min, dot_prod), state_max);
      *result_in_batch = static_cast<int16_t>(dot_prod);
      result_in_batch += n_memory;
    }
  }

  const int16_t* weights_time_data =
      GetTensorData<int16_t>(weights_time_tensor);
  for (int b = 0; b < n_batch; ++b) {
    rowinner_s32_s16xs16_Mmod2_Nmod4(
        weights_time_data, state_ptr + b * n_memory * n_filter,
        time_s32_scratch + b * n_filter, n_filter, n_memory);
  }

  if (bias_tensor != nullptr) {
    const int32_t* bias_data = GetTensorData<int32_t>(bias_tensor);
    for (int i = 0; i < n_batch; ++i) {
      int32_t* output_ptr = output_s32_scratch + i * n_unit;
      for (int j = 0; j < n_unit; ++j) {
        *output_ptr++ = bias_data[j];
      }
    }
  } else {
    memset(output_s32_scratch, 0, n_batch * n_unit * sizeof(int32_t));
  }

  for (int b = 0; b < n_batch; ++b) {
    int32_t* output_temp_ptr = output_s32_scratch + b * n_unit;
    int32_t* scratch_ptr_batch = time_s32_scratch + b * n_filter;
    for (int i = 0; i < n_unit; ++i) {
      for (int j = 0; j < n_rank; ++j) {
        output_temp_ptr[i] += *scratch_ptr_batch++;
      }
    }
  }

  const int32_t output_max = std::numeric_limits<int8_t>::max();
  const int32_t output_min = std::numeric_limits<int8_t>::min();
  int8_t* output_data = GetTensorData<int8_t>(output_tensor);
  for (int i = 0; i < n_batch * n_unit; ++i) {
    int32_t x1 = output_s32_scratch[i];
    int32_t x2 = MultiplyByQuantizedMultiplier(
        x1, data.reference_op_data.effective_scale_2_a,
        data.reference_op_data.effective_scale_2_b);
    int32_t x3 = x2 + data.reference_op_data.output_zero_point;
    int32_t x4 = std::min(std::max(output_min, x3), output_max);
    output_data[i] = static_cast<int8_t>(x4);
  }
}

}  // namespace

TfLiteStatus HexagonSvdfEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const HexagonOpDataSvdf& data =
      *(static_cast<const HexagonOpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input = GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5) ? GetEvalInput(context, node, kSvdfBiasTensor)
                             : nullptr;
  TfLiteEvalTensor* activation_state =
      GetMutableEvalInput(context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output = GetEvalOutput(context, node, kSvdfOutputTensor);

  if (data.optimizable != 0) {
    HexagonSvdfOptimizedEvalInt8(context, input, weights_feature, weights_time,
                                 bias, params, activation_state, output, data);
  } else {
    EvalInt16SvdfReference(context, node, input, weights_feature, weights_time,
                           bias, params, activation_state, output,
                           data.reference_op_data);
  }
  return kTfLiteOk;
}

void* HexagonSvdfInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(HexagonOpDataSvdf));
}

TfLiteStatus HexagonSvdfPrepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus prepare_status = PrepareSvdf(context, node);
  if (prepare_status != kTfLiteOk) {
    return prepare_status;
  }

  HexagonOpDataSvdf* data = static_cast<HexagonOpDataSvdf*>(node->user_data);
  HexagonSvdfOptimizationEvaluation(context, node, data);

  if (data->optimizable != 0) {
    TF_LITE_ENSURE_OK(context,
                      HexagonSvdfOptimizedPrepare(context, node, data));
  }

  return kTfLiteOk;
}

TFLMRegistration Register_SVDF_INT8() {
  return RegisterOp(HexagonSvdfInit, HexagonSvdfPrepare, HexagonSvdfEvalInt8);
}

const TfLiteEvalTensor*
HexagonLegacyGetEvalInput(const TfLiteContext* context, const TfLiteNode* node, int index) asm(
    "_ZN6tflite5micro12GetEvalInputEPK13TfLiteContextPK10TfLiteNodei");
__attribute__((weak, used)) const TfLiteEvalTensor* HexagonLegacyGetEvalInput(
    const TfLiteContext* context, const TfLiteNode* node, int index) {
  return GetEvalInput(context, node, index);
}

TfLiteEvalTensor*
HexagonLegacyGetEvalOutput(const TfLiteContext* context, const TfLiteNode* node, int index) asm(
    "_ZN6tflite5micro13GetEvalOutputEPK13TfLiteContextPK10TfLiteNodei");
__attribute__((weak, used)) TfLiteEvalTensor* HexagonLegacyGetEvalOutput(
    const TfLiteContext* context, const TfLiteNode* node, int index) {
  return GetEvalOutput(context, node, index);
}

TfLiteEvalTensor*
HexagonLegacyGetMutableEvalInput(const TfLiteContext* context, const TfLiteNode* node, int index) asm(
    "_ZN6tflite5micro19GetMutableEvalInputEPK13TfLiteContextPK10TfLiteNodei");
__attribute__((weak, used)) TfLiteEvalTensor* HexagonLegacyGetMutableEvalInput(
    const TfLiteContext* context, const TfLiteNode* node, int index) {
  return GetMutableEvalInput(context, node, index);
}

}  // namespace micro
}  // namespace tflite
