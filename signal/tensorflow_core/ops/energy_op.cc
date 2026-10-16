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

#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"

namespace tensorflow {
namespace signal {
namespace {

void EnergyShape(TF_ShapeInferenceContext* ctx, TF_Status* status) {
  TF_ShapeHandle* in0 = TF_NewShapeHandle();
  TF_ShapeHandle* ranked = TF_NewShapeHandle();
  TF_DimensionHandle* dim0 = TF_NewDimensionHandle();

  TF_ShapeInferenceContextGetInput(ctx, 0, in0, status);
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextWithRank(ctx, in0, 1, ranked, status);
  }
  if (TF_GetCode(status) == TF_OK) {
    TF_ShapeInferenceContextDim(ctx, ranked, 0, dim0);
    if (TF_DimensionHandleValueKnown(dim0)) {
      int64_t length = TF_DimensionHandleValue(dim0) / 2;
      TF_ShapeHandle* out = TF_ShapeInferenceContextVectorFromSize(
          ctx, static_cast<size_t>(length));
      TF_ShapeInferenceContextSetOutput(ctx, 0, out, status);
      TF_DeleteShapeHandle(out);
    } else {
      TF_ShapeInferenceContextSetUnknownShape(ctx, status);
    }
  }

  TF_DeleteDimensionHandle(dim0);
  TF_DeleteShapeHandle(ranked);
  TF_DeleteShapeHandle(in0);
}

// Calculate the energy of a spectral array. Only elements in the index range
// [start_index, end_index] are calculated, and the rest are set to zero.
//
// input: A 1-D frequency domain signal of complex int16 elements
// output: A 1-D array of uint32 elements containing the square of the absolute
//         value of each element in input.
//
// start_index: index in input to start calculating from. Default: 0
// end_index: last index in the input to calculate. Default: last element of
// input
// TODO(b/286250473): change back name after name clash resolved
__attribute__((constructor)) void RegisterSignalEnergyOp() {
  TF_Status* status = TF_NewStatus();
  TF_OpDefinitionBuilder* builder = TF_NewOpDefinitionBuilder("SignalEnergy");
  TF_OpDefinitionBuilderAddAttr(builder, "start_index: int");
  TF_OpDefinitionBuilderAddAttr(builder, "end_index: int");
  TF_OpDefinitionBuilderAddInput(builder, "input: int16");
  TF_OpDefinitionBuilderAddOutput(builder, "output: uint32");
  TF_OpDefinitionBuilderSetShapeInferenceFunction(builder, &EnergyShape);
  TF_RegisterOpDefinition(builder, status);
  TF_DeleteStatus(status);
}

}  // namespace
}  // namespace signal
}  // namespace tensorflow
