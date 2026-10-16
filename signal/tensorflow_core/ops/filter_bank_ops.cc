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

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

bool CheckSame1DDim(TF_ShapeInferenceContext* ctx, TF_ShapeHandle* first,
                    TF_ShapeHandle* second, TF_Status* status) {
  TF_DimensionHandle* dim0 = TF_NewDimensionHandle();
  TF_DimensionHandle* dim1 = TF_NewDimensionHandle();
  TF_ShapeInferenceContextDim(ctx, first, 0, dim0);
  TF_ShapeInferenceContextDim(ctx, second, 0, dim1);
  if (TF_DimensionHandleValueKnown(dim0) &&
      TF_DimensionHandleValueKnown(dim1)) {
    int64_t v0 = TF_DimensionHandleValue(dim0);
    int64_t v1 = TF_DimensionHandleValue(dim1);
    if (v0 != v1) {
      char msg[128];
      std::snprintf(msg, sizeof(msg), "Dimension must be %lld but is %lld",
                    static_cast<long long>(v0), static_cast<long long>(v1));
      TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
    }
  }
  TF_DeleteDimensionHandle(dim1);
  TF_DeleteDimensionHandle(dim0);
  return TF_GetCode(status) == TF_OK;
}

void FilterBankShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  int num_channels = 0;
  if (!GetAttrInt(ctx, "num_channels", &num_channels, status)) {
    return;
  }

  TF_ShapeHandle* inputs[6];
  TF_ShapeHandle* ranked[6];
  for (int i = 0; i < 6; ++i) {
    inputs[i] = TF_NewShapeHandle();
    ranked[i] = TF_NewShapeHandle();
  }

  for (int i = 0; i < 6 && TF_GetCode(status) == TF_OK; ++i) {
    TF_ShapeInferenceContextGetInput(ctx, i, inputs[i], status);
    if (TF_GetCode(status) == TF_OK) {
      TF_ShapeInferenceContextWithRank(ctx, inputs[i], 1, ranked[i], status);
    }
  }

  if (TF_GetCode(status) == TF_OK) {
    // Inputs 1,2 must have the same shape.
    CheckSame1DDim(ctx, ranked[1], ranked[2], status);
  }
  if (TF_GetCode(status) == TF_OK) {
    // Inputs 3,4,5 must have the same shape (num_channels + 1).
    CheckDimValue(ctx, ranked[3], 0, num_channels + 1, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    CheckDimValue(ctx, ranked[4], 0, num_channels + 1, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    CheckDimValue(ctx, ranked[5], 0, num_channels + 1, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeHandle* out = TF_ShapeInferenceContextVectorFromSize(
        ctx, static_cast<size_t>(num_channels));
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
    TF_DeleteShapeHandle(out);
  }

  for (int i = 0; i < 6; ++i) {
    TF_DeleteShapeHandle(ranked[i]);
    TF_DeleteShapeHandle(inputs[i]);
  }
}

void FilterBank1DIdentityShape(TF_ShapeInferenceContext* ctx,
                               TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in0);
}

void FilterBankSpectralSubtractionShape(TF_ShapeInferenceContext* ctx,
                                        TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 1, out, status);
  }
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in0);
}

// Applies a mel filter bank of size num_channels to an input spectral energy
// array See filter_bank_ops.py for how weights, unweights,
// channel_frequency_start, channel_weight_start, channel_widths are
// pre-calculated.
//
// Applies a square root to each element in the input then shift right by
// scale_bits before writing the result to output.
//
// Applies spectral subtraction to a filter bank output of size num_channels.
// Outputs the clean filter bank channels and the noise estimate for each
// channel.
//
// Applies natural log to each element in input with pre-shift and post scaling.
// TODO(b/286250473): change back name after name clash resolved
void RegisterSignalFilterBankOps() {
  TF_Status* status = TF_NewStatus();

  TF_OpDefinitionBuilder* fb_builder =
      TF_NewOpDefinitionBuilder("SignalFilterBank");
  TF_OpDefinitionBuilderAddAttr(fb_builder, "num_channels: int >= 0");
  TF_OpDefinitionBuilderAddInput(fb_builder, "input: uint32");
  TF_OpDefinitionBuilderAddInput(fb_builder, "weights: int16");
  TF_OpDefinitionBuilderAddInput(fb_builder, "unweights: int16");
  TF_OpDefinitionBuilderAddInput(fb_builder, "channel_frequency_starts: int16");
  TF_OpDefinitionBuilderAddInput(fb_builder, "channel_weight_starts: int16");
  TF_OpDefinitionBuilderAddInput(fb_builder, "channel_widths: int16");
  TF_OpDefinitionBuilderAddOutput(fb_builder, "output: uint64");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(fb_builder, &FilterBankShape);
  TF_RegisterOpDefinition(fb_builder, status);

  TF_OpDefinitionBuilder* sqrt_builder =
      TF_NewOpDefinitionBuilder("SignalFilterBankSquareRoot");
  TF_OpDefinitionBuilderAddInput(sqrt_builder, "input: uint64");
  TF_OpDefinitionBuilderAddInput(sqrt_builder, "scale_bits: int32");
  TF_OpDefinitionBuilderAddOutput(sqrt_builder, "output: uint32");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(sqrt_builder,
                                                  &FilterBank1DIdentityShape);
  TF_RegisterOpDefinition(sqrt_builder, status);

  TF_OpDefinitionBuilder* sub_builder =
      TF_NewOpDefinitionBuilder("SignalFilterBankSpectralSubtraction");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "num_channels: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "smoothing: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "one_minus_smoothing: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "alternate_smoothing: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder,
                                "alternate_one_minus_smoothing: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "smoothing_bits: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "min_signal_remaining: int >= 0");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "clamping: bool");
  TF_OpDefinitionBuilderAddAttr(sub_builder, "spectral_subtraction_bits: int");
  TF_OpDefinitionBuilderAddInput(sub_builder, "input: uint32");
  TF_OpDefinitionBuilderAddOutput(sub_builder, "output: uint32");
  TF_OpDefinitionBuilderAddOutput(sub_builder, "noise_estimate: uint32");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(
      sub_builder, &FilterBankSpectralSubtractionShape);
  TF_RegisterOpDefinition(sub_builder, status);

  TF_OpDefinitionBuilder* log_builder =
      TF_NewOpDefinitionBuilder("SignalFilterBankLog");
  TF_OpDefinitionBuilderAddAttr(log_builder, "output_scale: int >= 1");
  TF_OpDefinitionBuilderAddAttr(log_builder, "input_correction_bits: int >= 0");
  TF_OpDefinitionBuilderAddInput(log_builder, "input: uint32");
  TF_OpDefinitionBuilderAddOutput(log_builder, "output: int16");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(log_builder,
                                                  &FilterBank1DIdentityShape);
  TF_RegisterOpDefinition(log_builder, status);

  TF_DeleteStatus(status);
}

const bool kSignalFilterBankOpsRegistered = []() {
  RegisterSignalFilterBankOps();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
