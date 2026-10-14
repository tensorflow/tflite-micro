/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_TESTING_TEST_ALL_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_TESTING_TEST_ALL_OP_RESOLVER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {

// Number of all built-in operators currently supported by TFLM.
constexpr unsigned int kBuiltinOpsCount = 104;

// Populates all built-in operators into any MicroMutableOpResolver.
template <unsigned int tOpCount>
TfLiteStatus RegisterBuiltinOps(MicroMutableOpResolver<tOpCount>& resolver) {
  static_assert(tOpCount >= kBuiltinOpsCount,
                "Op resolver capacity too small for all built-in ops");
  TF_LITE_ENSURE_STATUS(resolver.AddAbs());
  TF_LITE_ENSURE_STATUS(resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(resolver.AddAddN());
  TF_LITE_ENSURE_STATUS(resolver.AddArgMax());
  TF_LITE_ENSURE_STATUS(resolver.AddArgMin());
  TF_LITE_ENSURE_STATUS(resolver.AddAssignVariable());
  TF_LITE_ENSURE_STATUS(resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddBatchMatMul());
  TF_LITE_ENSURE_STATUS(resolver.AddBatchToSpaceNd());
  TF_LITE_ENSURE_STATUS(resolver.AddBroadcastArgs());
  TF_LITE_ENSURE_STATUS(resolver.AddBroadcastTo());
  TF_LITE_ENSURE_STATUS(resolver.AddCallOnce());
  TF_LITE_ENSURE_STATUS(resolver.AddCast());
  TF_LITE_ENSURE_STATUS(resolver.AddCeil());
  TF_LITE_ENSURE_STATUS(resolver.AddCircularBuffer());
  TF_LITE_ENSURE_STATUS(resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(resolver.AddCos());
  TF_LITE_ENSURE_STATUS(resolver.AddCumSum());
  TF_LITE_ENSURE_STATUS(resolver.AddDecode());
  TF_LITE_ENSURE_STATUS(resolver.AddDepthToSpace());
  TF_LITE_ENSURE_STATUS(resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(resolver.AddDetectionPostprocess());
  TF_LITE_ENSURE_STATUS(resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(resolver.AddDynamicUpdateSlice());
  TF_LITE_ENSURE_STATUS(resolver.AddElu());
  TF_LITE_ENSURE_STATUS(resolver.AddEmbeddingLookup());
  TF_LITE_ENSURE_STATUS(resolver.AddEqual());
  TF_LITE_ENSURE_STATUS(resolver.AddEthosU());
  TF_LITE_ENSURE_STATUS(resolver.AddExp());
  TF_LITE_ENSURE_STATUS(resolver.AddExpandDims());
  TF_LITE_ENSURE_STATUS(resolver.AddFill());
  TF_LITE_ENSURE_STATUS(resolver.AddFloor());
  TF_LITE_ENSURE_STATUS(resolver.AddFloorDiv());
  TF_LITE_ENSURE_STATUS(resolver.AddFloorMod());
  TF_LITE_ENSURE_STATUS(resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(resolver.AddGather());
  TF_LITE_ENSURE_STATUS(resolver.AddGatherNd());
  TF_LITE_ENSURE_STATUS(resolver.AddGreater());
  TF_LITE_ENSURE_STATUS(resolver.AddGreaterEqual());
  TF_LITE_ENSURE_STATUS(resolver.AddHardSwish());
  TF_LITE_ENSURE_STATUS(resolver.AddIf());
  TF_LITE_ENSURE_STATUS(resolver.AddL2Normalization());
  TF_LITE_ENSURE_STATUS(resolver.AddL2Pool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddLeakyRelu());
  TF_LITE_ENSURE_STATUS(resolver.AddLess());
  TF_LITE_ENSURE_STATUS(resolver.AddLessEqual());
  TF_LITE_ENSURE_STATUS(resolver.AddLog());
  TF_LITE_ENSURE_STATUS(resolver.AddLogSoftmax());
  TF_LITE_ENSURE_STATUS(resolver.AddLogicalAnd());
  TF_LITE_ENSURE_STATUS(resolver.AddLogicalNot());
  TF_LITE_ENSURE_STATUS(resolver.AddLogicalOr());
  TF_LITE_ENSURE_STATUS(resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(resolver.AddMean());
  TF_LITE_ENSURE_STATUS(resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(resolver.AddMirrorPad());
  TF_LITE_ENSURE_STATUS(resolver.AddMul());
  TF_LITE_ENSURE_STATUS(resolver.AddNeg());
  TF_LITE_ENSURE_STATUS(resolver.AddNotEqual());
  TF_LITE_ENSURE_STATUS(resolver.AddPack());
  TF_LITE_ENSURE_STATUS(resolver.AddPad());
  TF_LITE_ENSURE_STATUS(resolver.AddPadV2());
  TF_LITE_ENSURE_STATUS(resolver.AddPrelu());
  TF_LITE_ENSURE_STATUS(resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(resolver.AddReadVariable());
  TF_LITE_ENSURE_STATUS(resolver.AddReduceAll());
  TF_LITE_ENSURE_STATUS(resolver.AddReduceMax());
  TF_LITE_ENSURE_STATUS(resolver.AddReduceMin());
  TF_LITE_ENSURE_STATUS(resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(resolver.AddRelu6());
  TF_LITE_ENSURE_STATUS(resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(resolver.AddResizeBilinear());
  TF_LITE_ENSURE_STATUS(resolver.AddResizeNearestNeighbor());
  TF_LITE_ENSURE_STATUS(resolver.AddReverseV2());
  TF_LITE_ENSURE_STATUS(resolver.AddRound());
  TF_LITE_ENSURE_STATUS(resolver.AddRsqrt());
  TF_LITE_ENSURE_STATUS(resolver.AddSelectV2());
  TF_LITE_ENSURE_STATUS(resolver.AddShape());
  TF_LITE_ENSURE_STATUS(resolver.AddSin());
  TF_LITE_ENSURE_STATUS(resolver.AddSlice());
  TF_LITE_ENSURE_STATUS(resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(resolver.AddSpaceToBatchNd());
  TF_LITE_ENSURE_STATUS(resolver.AddSpaceToDepth());
  TF_LITE_ENSURE_STATUS(resolver.AddSplit());
  TF_LITE_ENSURE_STATUS(resolver.AddSplitV());
  TF_LITE_ENSURE_STATUS(resolver.AddSqrt());
  TF_LITE_ENSURE_STATUS(resolver.AddSquare());
  TF_LITE_ENSURE_STATUS(resolver.AddSquaredDifference());
  TF_LITE_ENSURE_STATUS(resolver.AddSqueeze());
  TF_LITE_ENSURE_STATUS(resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(resolver.AddSub());
  TF_LITE_ENSURE_STATUS(resolver.AddSum());
  TF_LITE_ENSURE_STATUS(resolver.AddSvdf());
  TF_LITE_ENSURE_STATUS(resolver.AddTanh());
  TF_LITE_ENSURE_STATUS(resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(resolver.AddTransposeConv());
  TF_LITE_ENSURE_STATUS(resolver.AddUnidirectionalSequenceLSTM());
  TF_LITE_ENSURE_STATUS(resolver.AddUnpack());
  TF_LITE_ENSURE_STATUS(resolver.AddVarHandle());
  TF_LITE_ENSURE_STATUS(resolver.AddWhile());
  TF_LITE_ENSURE_STATUS(resolver.AddZerosLike());
  return kTfLiteOk;
}

// WARNING: TestAllOpResolver is intended EXCLUSIVELY for unit tests, model
// evaluation, benchmarks, and development tooling.
//
// Do NOT use this class in production microcontroller firmware!
//
// Instantiating TestAllOpResolver statically links every single built-in
// operator kernel into the binary, resulting in massive Flash/ROM and RAM
// code bloat that will exceed microcontroller storage limits.
//
// For production deployments, use MicroMutableOpResolver with only the specific
// operators your model requires:
//   tflite::MicroMutableOpResolver<3> resolver;
//   resolver.AddConv2D();
//   resolver.AddRelu();
//   resolver.AddFullyConnected();
//
// Usage for testing:
//   // Built-in ops only:
//   tflite::TestAllOpResolver<> resolver;
//
//   // Built-in ops + 15 custom op slots:
//   tflite::TestAllOpResolver<15> resolver;
//   resolver.AddCustom("MyCustomOp", Register_MY_CUSTOM_OP());
template <unsigned int tExtraOpCount = 0>
class TestAllOpResolver
    : public MicroMutableOpResolver<kBuiltinOpsCount + tExtraOpCount> {
 public:
  TestAllOpResolver() { RegisterBuiltinOps(*this); }

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TESTING_TEST_ALL_OP_RESOLVER_H_
