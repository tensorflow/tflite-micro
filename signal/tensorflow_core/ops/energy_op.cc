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

void EnergyShape(TF_ShapeInferenceContext* raw_ctx, TF_Status* status) {
  ShapeInferenceContext ctx(raw_ctx, status);
  ShapeHandle in = ctx.WithRank(ctx.Input(0), 1);
  DimensionHandle dim0 = ctx.Dim(in, 0);
  if (ctx.ValueKnown(dim0)) {
    ctx.SetOutput(0, ctx.VectorFromSize(ctx.Value(dim0) / 2));
  } else {
    ctx.SetOutput(0, in);
  }
}

// Registers the SignalEnergy custom op (documented in energy_op.py).
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalEnergyOp() {
  SignalOpDefBuilder("SignalEnergy")
      .Attr("start_index: int")
      .Attr("end_index: int")
      .Input("input: int16")
      .Output("output: uint32")
      .SetShapeFn(&EnergyShape)
      .Register();
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
