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

#ifndef TFLM_BENCHMARK_OP_RESOLVER_H_
#define TFLM_BENCHMARK_OP_RESOLVER_H_

#include <memory>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"

namespace tflite {

using TflmOpResolver = MicroMutableOpResolver<113>;

inline TfLiteStatus CreateOpResolver(TflmOpResolver& op_resolver) {
  TF_LITE_ENSURE_STATUS(op_resolver.AddAbs());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAddN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddArgMax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddArgMin());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAssignVariable());
  TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBatchMatMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBatchToSpaceNd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBroadcastArgs());
  TF_LITE_ENSURE_STATUS(op_resolver.AddBroadcastTo());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCallOnce());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCeil());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCircularBuffer());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
  TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCos());
  TF_LITE_ENSURE_STATUS(op_resolver.AddCumSum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDelay());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthToSpace());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDetectionPostprocess());
  TF_LITE_ENSURE_STATUS(op_resolver.AddDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddElu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEmbeddingLookup());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEnergy());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEqual());
  TF_LITE_ENSURE_STATUS(op_resolver.AddEthosU());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExp());
  TF_LITE_ENSURE_STATUS(op_resolver.AddExpandDims());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFftAutoScale());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFill());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBank());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankLog());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSpectralSubtraction());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFilterBankSquareRoot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFloor());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFloorDiv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFloorMod());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFramer());
  TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
  TF_LITE_ENSURE_STATUS(op_resolver.AddGather());
  TF_LITE_ENSURE_STATUS(op_resolver.AddGatherNd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddGreater());
  TF_LITE_ENSURE_STATUS(op_resolver.AddGreaterEqual());
  TF_LITE_ENSURE_STATUS(op_resolver.AddHardSwish());
  TF_LITE_ENSURE_STATUS(op_resolver.AddIf());
  TF_LITE_ENSURE_STATUS(op_resolver.AddIrfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddL2Normalization());
  TF_LITE_ENSURE_STATUS(op_resolver.AddL2Pool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLeakyRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLess());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLessEqual());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLog());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogicalAnd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogicalNot());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogicalOr());
  TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMaximum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMinimum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMirrorPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
  TF_LITE_ENSURE_STATUS(op_resolver.AddNeg());
  TF_LITE_ENSURE_STATUS(op_resolver.AddNotEqual());
  TF_LITE_ENSURE_STATUS(op_resolver.AddOverlapAdd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPCAN());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPack());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPadV2());
  TF_LITE_ENSURE_STATUS(op_resolver.AddPrelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReadVariable());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReduceMax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRelu6());
  TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeBilinear());
  TF_LITE_ENSURE_STATUS(op_resolver.AddResizeNearestNeighbor());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRfft());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRound());
  TF_LITE_ENSURE_STATUS(op_resolver.AddRsqrt());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSelectV2());
  TF_LITE_ENSURE_STATUS(op_resolver.AddShape());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSin());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSpaceToBatchNd());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSpaceToDepth());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSplit());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSplitV());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSqrt());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSquare());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSquaredDifference());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSqueeze());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStacker());
  TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSum());
  TF_LITE_ENSURE_STATUS(op_resolver.AddSvdf());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
  TF_LITE_ENSURE_STATUS(op_resolver.AddTransposeConv());
  TF_LITE_ENSURE_STATUS(op_resolver.AddUnidirectionalSequenceLSTM());
  TF_LITE_ENSURE_STATUS(op_resolver.AddUnpack());
  TF_LITE_ENSURE_STATUS(op_resolver.AddVarHandle());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWhile());
  TF_LITE_ENSURE_STATUS(op_resolver.AddWindow());
  TF_LITE_ENSURE_STATUS(op_resolver.AddZerosLike());

  return kTfLiteOk;
}

}  // namespace tflite
#endif  // TFLM_BENCHMARK_OP_RESOLVER_H_
