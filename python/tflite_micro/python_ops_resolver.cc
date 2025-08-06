/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "python/tflite_micro/python_ops_resolver.h"

#include "tensorflow/lite/micro/kernels/micro_ops.h"

namespace tflite {

PythonOpsResolver::PythonOpsResolver() {
  // Please keep this list of Builtin Operators in alphabetical order.
  AddAbs();
  AddAdd();
  AddAddN();
  AddArgMax();
  AddArgMin();
  AddAssignVariable();
  AddAveragePool2D();
  AddBatchMatMul();
  AddBatchToSpaceNd();
  AddBroadcastArgs();
  AddBroadcastTo();
  AddCallOnce();
  AddCast();
  AddCeil();
  AddCircularBuffer();
  AddConcatenation();
  AddConv2D();
  AddCos();
  AddCumSum();
  AddDelay();
  AddDepthToSpace();
  AddDepthwiseConv2D();
  AddDequantize();
  AddDetectionPostprocess();
  AddDiv();
  AddElu();
  AddEmbeddingLookup();
  AddEnergy();
  AddEqual();
  AddEthosU();
  AddExp();
  AddExpandDims();
  AddFftAutoScale();
  AddFill();
  AddFilterBank();
  AddFilterBankLog();
  AddFilterBankSpectralSubtraction();
  AddFilterBankSquareRoot();
  AddFloor();
  AddFloorDiv();
  AddFloorMod();
  AddFramer();
  AddFullyConnected();
  AddGather();
  AddGatherNd();
  AddGreater();
  AddGreaterEqual();
  AddHardSwish();
  AddIf();
  AddIrfft();
  AddL2Normalization();
  AddL2Pool2D();
  AddLeakyRelu();
  AddLess();
  AddLessEqual();
  AddLog();
  AddLogSoftmax();
  AddLogicalAnd();
  AddLogicalNot();
  AddLogicalOr();
  AddLogistic();
  AddMaxPool2D();
  AddMaximum();
  AddMean();
  AddMinimum();
  AddMirrorPad();
  AddMul();
  AddNeg();
  AddNotEqual();
  AddOverlapAdd();
  AddPCAN();
  AddPack();
  AddPad();
  AddPadV2();
  AddPrelu();
  AddQuantize();
  AddReadVariable();
  AddReduceMax();
  AddReduceMin();
  AddRelu();
  AddRelu6();
  AddReshape();
  AddResizeBilinear();
  AddResizeNearestNeighbor();
  AddReverseV2();
  AddRfft();
  AddRound();
  AddRsqrt();
  AddSelectV2();
  AddShape();
  AddSin();
  AddSlice();
  AddSoftmax();
  AddSpaceToBatchNd();
  AddSpaceToDepth();
  AddSplit();
  AddSplitV();
  AddSqrt();
  AddSquare();
  AddSquaredDifference();
  AddSqueeze();
  AddStacker();
  AddStridedSlice();
  AddSub();
  AddSum();
  AddSvdf();
  AddTanh();
  AddTranspose();
  AddTransposeConv();
  AddUnidirectionalSequenceLSTM();
  AddUnpack();
  AddVarHandle();
  AddWhile();
  AddWindow();
  AddZerosLike();
}

}  // namespace tflite
