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

#include <cstdint>

#include "signal/tensorflow_core/ops/shape_inference_utils.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void RfftShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int fft_length = 0;
  if (!ctx.GetAttr("fft_length", &fft_length)) {
    return;
  }
  ShapeHandle in = ctx.WithRankAtLeast(ctx.Input(0), 1);
  int64_t dim =
      static_cast<int64_t>((fft_length / 2) + 1) * 2;  // * 2 for complex
  ctx.SetOutput(0, ctx.ReplaceLastDim(in, dim));
}

void IrfftShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  int fft_length = 0;
  if (!ctx.GetAttr("fft_length", &fft_length)) {
    return;
  }
  ShapeHandle in = ctx.WithRankAtLeast(ctx.Input(0), 1);
  ctx.SetOutput(0, ctx.ReplaceLastDim(in, fft_length));
}

void FftAutoScaleShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  ShapeHandle out = ctx.WithRank(ctx.Input(0), 1);
  ctx.SetOutput(0, out);
  ctx.SetOutput(1, ctx.Scalar());
}

// Registers the Signal FFT custom ops (documented in fft_ops.py).
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalFftOps() {
  SignalOpDefBuilder("SignalRfft")
      .Attr("T: {float, int16, int32}")
      .Attr("fft_length: int >= 2")
      .Input("input: T")
      .Output("output: T")
      .SetShapeFn(&RfftShape)
      .Register();

  SignalOpDefBuilder("SignalIrfft")
      .Attr("T: {float, int16, int32}")
      .Attr("fft_length: int >= 2")
      .Input("input: T")
      .Output("output: T")
      .SetShapeFn(&IrfftShape)
      .Register();

  SignalOpDefBuilder("SignalFftAutoScale")
      .Input("input: int16")
      .Output("output: int16")
      .Output("scale_bits: int32")
      .SetShapeFn(&FftAutoScaleShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
