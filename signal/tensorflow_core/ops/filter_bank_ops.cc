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

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void FilterBankShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int num_channels = 0;
  if (!ctx.GetAttr("num_channels", &num_channels)) {
    return;
  }

  ctx.WithRank(ctx.Input(0), 1);

  // Inputs 1 and 2 must be 1-D with the same dimension.
  ShapeHandle in1 = ctx.WithRank(ctx.Input(1), 1);
  ShapeHandle in2 = ctx.WithRank(ctx.Input(2), 1);
  DimensionHandle dim1 = ctx.Dim(in1, 0);
  if (ctx.ValueKnown(dim1)) {
    ctx.CheckDim(in2, 0, ctx.Value(dim1));
  }

  // Inputs 3, 4, and 5 must be 1-D with dimension (num_channels + 1).
  for (int i = 3; i <= 5; ++i) {
    ShapeHandle in = ctx.WithRank(ctx.Input(i), 1);
    ctx.CheckDim(in, 0, num_channels + 1);
  }

  ctx.SetOutput(0, ctx.VectorFromSize(num_channels));
}

void FilterBank1DIdentityShape(TF_ShapeInferenceContext* raw_ctx,
                               TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  ShapeHandle out = ctx.WithRank(ctx.Input(0), 1);
  ctx.SetOutput(0, out);
}

void FilterBankSpectralSubtractionShape(TF_ShapeInferenceContext* raw_ctx,
                                        TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  ShapeHandle out = ctx.WithRank(ctx.Input(0), 1);
  ctx.SetOutput(0, out);
  ctx.SetOutput(1, out);
}

// Registers Signal filter bank custom ops (documented in filter_bank_ops.py).
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFilterBankOps() {
  SignalOpDefBuilder("SignalFilterBank")
      .Attr("num_channels: int >= 0")
      .Input("input: uint32")
      .Input("weights: int16")
      .Input("unweights: int16")
      .Input("channel_frequency_starts: int16")
      .Input("channel_weight_starts: int16")
      .Input("channel_widths: int16")
      .Output("output: uint64")
      .SetShapeFn(&FilterBankShape)
      .Register();

  SignalOpDefBuilder("SignalFilterBankSquareRoot")
      .Input("input: uint64")
      .Input("scale_bits: int32")
      .Output("output: uint32")
      .SetShapeFn(&FilterBank1DIdentityShape)
      .Register();

  SignalOpDefBuilder("SignalFilterBankSpectralSubtraction")
      .Attr("num_channels: int >= 0")
      .Attr("smoothing: int >= 0")
      .Attr("one_minus_smoothing: int >= 0")
      .Attr("alternate_smoothing: int >= 0")
      .Attr("alternate_one_minus_smoothing: int >= 0")
      .Attr("smoothing_bits: int >= 0")
      .Attr("min_signal_remaining: int >= 0")
      .Attr("clamping: bool")
      .Attr("spectral_subtraction_bits: int")
      .Input("input: uint32")
      .Output("output: uint32")
      .Output("noise_estimate: uint32")
      .SetShapeFn(&FilterBankSpectralSubtractionShape)
      .Register();

  SignalOpDefBuilder("SignalFilterBankLog")
      .Attr("output_scale: int >= 1")
      .Attr("input_correction_bits: int >= 0")
      .Input("input: uint32")
      .Output("output: int16")
      .SetShapeFn(&FilterBank1DIdentityShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
