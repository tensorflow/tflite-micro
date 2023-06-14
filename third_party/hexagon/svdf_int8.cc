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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "third_party/hexagon/hexagon_svdf.h"
#include "third_party/hexagon/hexagon_tflm_translation_svdf.h"

namespace tflite {

TfLiteStatus HexagonSvdfEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const HexagonOpDataSvdf& data =
      *(static_cast<const HexagonOpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  if (tflite::hexagon_svdf::HexagonOptimizable(context, node)) {
    tflite::hexagon_svdf::HexagonEvalIntegerSVDF(
        context, node, input, weights_feature, weights_time, bias, params,
        activation_state, output, node->user_data);
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
  void* data = context->AllocatePersistentBuffer(context, sizeof(OpDataSvdf));

  if (data == nullptr) {
    return nullptr;
  }

  HexagonOpDataSvdf* opdata = static_cast<HexagonOpDataSvdf*>(data);
  opdata->hexagon_data =
      tflite::hexagon_svdf::HexagonInit(context, buffer, length);

  return data;
}

TfLiteStatus HexagonSvdfPrepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteStatus prepare_status = PrepareSvdf(context, node);
  if (prepare_status != kTfLiteOk) {
    return prepare_status;
  }

  tflite::hexagon_svdf::HexagonOptimizationEvaluation(context, node);

  if (tflite::hexagon_svdf::HexagonOptimizable(context, node)) {
    TF_LITE_ENSURE_OK(context,
                      tflite::hexagon_svdf::HexagonPrepare(context, node));
  }

  return kTfLiteOk;
}

TFLMRegistration Register_SVDF_INT8() {
  return tflite::micro::RegisterOp(HexagonSvdfInit, HexagonSvdfPrepare,
                                   HexagonSvdfEvalInt8);
}

}  // namespace tflite
