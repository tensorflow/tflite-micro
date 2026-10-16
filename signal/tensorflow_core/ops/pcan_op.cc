/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void PcanShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* in1 = TF_NewShapeHandle();
  TF_ShapeHandle* in2 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeHandle* lut = TF_NewShapeHandle();

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextGetInput(ctx, 1, in1, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in1, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextGetInput(ctx, 2, in2, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in2, 1, lut, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }

  TF_DeleteShapeHandle(lut);
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in2);
  TF_DeleteShapeHandle(in1);
  TF_DeleteShapeHandle(in0);
}

// Determines whether per-channel amplitude-normalized (PCAN) auto gain control
// is applied, using either floating-point or fixed-point computation. If
// enabled, the dynamic range of the filterbank output is compressed by dividing
// by a power of the noise estimate.
//
// input: A 1-D array of mel-spectrum subband filter bank outputs.
// noise_estimate: A 1-D array of mel-spectrun subbabd noise estimates.
// gain_lut: A 1-D lookup table for gain calculation.
// output: A 1-D array of processed subband filter bank.
// snr_shift: Amount of right shift when calculating the SNR.
void RegisterSignalPcanOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalPCAN");
  TF_OpDefinitionBuilderAddAttr(builder, "snr_shift: int");
  TF_OpDefinitionBuilderAddInput(builder, "input: uint32");
  TF_OpDefinitionBuilderAddInput(builder, "noise_estimate: uint32");
  TF_OpDefinitionBuilderAddInput(builder, "gain_lut: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: uint32");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &PcanShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

const bool kSignalPcanOpRegistered = []() {
  RegisterSignalPcanOp();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
