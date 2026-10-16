/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void WindowShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* in1 = TF_NewShapeHandle();
  TF_ShapeHandle* out = TF_NewShapeHandle();
  TF_DimensionHandle* dim_in = TF_NewDimensionHandle();
  TF_DimensionHandle* dim_weights = TF_NewDimensionHandle();

  TF_ShapeInferenceContextGetInput(ctx, 1, in1, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in1, 1, out, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRankAtLeast(ctx, in0, 1, out, status);
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextDim(ctx, out, -1, dim_in);
    TF_ShapeInferenceContextDim(ctx, in1, 0, dim_weights);
    if (TF_DimensionHandleValueKnown(dim_weights)) {
      int64_t w_val = TF_DimensionHandleValue(dim_weights);
      int64_t in_val = TF_DimensionHandleValueKnown(dim_in)
                           ? TF_DimensionHandleValue(dim_in)
                           : -1;
      if (w_val != in_val) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "Dimension must be %lld but is %lld",
                      static_cast<long long>(in_val),
                      static_cast<long long>(w_val));
        TF_SetStatus(status, TF_INVALID_ARGUMENT, msg);
      }
    }
  }

  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
  }

  TF_DeleteDimensionHandle(dim_weights);
  TF_DeleteDimensionHandle(dim_in);
  TF_DeleteShapeHandle(out);
  TF_DeleteShapeHandle(in1);
  TF_DeleteShapeHandle(in0);
}

// TODO(b/286250473): change back name to "Window" after name clash resolved
void RegisterSignalWindowOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalWindow");
  TF_OpDefinitionBuilderAddAttr(builder, "shift: int");
  TF_OpDefinitionBuilderAddInput(builder, "input: int16");
  TF_OpDefinitionBuilderAddInput(builder, "weights: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: int16");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &WindowShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

const bool kSignalWindowOpRegistered = []() {
  RegisterSignalWindowOp();
  return true;
}();

}  // namespace
}  // namespace signal
}  // namespace tensorflow
