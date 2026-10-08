/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_H_
#define TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_H_

// These functions transform codes and data structures that are defined in the
// flatbuffer serialization format into in-memory values that are used by the
// runtime API and interpreter.

#include <cstddef>
#include <new>
#include <type_traits>

#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace micro {

// Interface class for builtin data allocations.
class BuiltinDataAllocator {
 public:
  virtual void* Allocate(size_t size, size_t alignment_hint) = 0;
  virtual void Deallocate(void* data) = 0;

  // Allocate a structure, but make sure it is a POD structure that doesn't
  // require constructors to run. The reason we do this, is that Interpreter's C
  // extension part will take ownership so destructors will not be run during
  // deallocation.
  template <typename T>
  T* AllocatePOD() {
    static_assert(std::is_trivially_destructible<T>::value,
                  "Builtin data structure must be POD.");
    void* allocated_memory = this->Allocate(sizeof(T), alignof(T));
    if (allocated_memory == nullptr) {
      return nullptr;
    }
    return new (allocated_memory) T();
  }

  virtual ~BuiltinDataAllocator() {}
};

using TfLiteBridgeBuiltinDataAllocator = BuiltinDataAllocator;

using TfLiteBridgeBuiltinParseFunction = TfLiteStatus (*)(
    const Operator* op, BuiltinDataAllocator* allocator, void** builtin_data);

// Converts the tensor data type used in the flat buffer to the representation
// used by the runtime.
TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type);

// CallBuiltinParseFunction is a wrapper function to wrap the parser function
// calls to Call parser(op, allocator, builtin_data)
TfLiteStatus CallBuiltinParseFunction(TfLiteBridgeBuiltinParseFunction parser,
                                      const Operator* op,
                                      BuiltinDataAllocator* allocator,
                                      void** builtin_data);

TfLiteStatus ParseAbs(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseAdd(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseAddN(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseArgMax(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseArgMin(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseAssignVariable(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseBatchMatMul(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

TfLiteStatus ParseBatchToSpaceNd(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseBroadcastArgs(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseBroadcastTo(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data);

TfLiteStatus ParseCallOnce(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseCeil(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseCast(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseConcatenation(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseConv2D(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseCos(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseCumsum(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseDepthToSpace(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseDequantize(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseDiv(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseDynamicUpdateSlice(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data);

TfLiteStatus ParseElu(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseEmbeddingLookup(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseEqual(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseExp(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseExpandDims(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseFill(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseFloor(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseFloorDiv(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseFloorMod(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseFullyConnected(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseGather(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseGatherNd(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseGreater(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseGreaterEqual(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseHardSwish(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseIf(const Operator* op, BuiltinDataAllocator* allocator,
                     void** builtin_data);

TfLiteStatus ParseL2Normalization(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseLeakyRelu(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLess(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseLessEqual(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLog(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseLogicalAnd(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLogicalNot(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLogicalOr(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseLogistic(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseLogSoftmax(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseLSTM(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseMaximum(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseMinimum(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseMirrorPad(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseMul(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseNeg(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseNotEqual(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParsePack(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParsePad(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParsePadV2(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParsePool(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParsePow(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParsePrelu(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseQuantize(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseReadVariable(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseReducer(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseRelu(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseRelu6(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseReshape(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseResizeBilinear(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseResizeNearestNeighbor(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data);

TfLiteStatus ParseReverseV2(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseRound(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseRsqrt(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseSelect(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseSelectV2(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data);

TfLiteStatus ParseShape(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseSin(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseSlice(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseSoftmax(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseSpaceToBatchNd(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data);

TfLiteStatus ParseSpaceToDepth(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseSplit(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseSplitV(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseSqueeze(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data);

TfLiteStatus ParseSqrt(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseSquare(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseSquaredDifference(const Operator* op,
                                    BuiltinDataAllocator* allocator,
                                    void** builtin_data);

TfLiteStatus ParseStridedSlice(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseSub(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data);

TfLiteStatus ParseSvdf(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseTanh(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data);

TfLiteStatus ParseTranspose(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseTransposeConv(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

TfLiteStatus ParseUnpack(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data);

TfLiteStatus ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data);

TfLiteStatus ParseVarHandle(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseWhile(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data);

TfLiteStatus ParseZerosLike(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data);

TfLiteStatus ParseBitwiseXor(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseRightShift(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data);

TfLiteStatus ParseStablehloScatter(const Operator* op,
                                   BuiltinDataAllocator* allocator,
                                   void** builtin_data);

TfLiteStatus ParseStablehloRngBitGenerator(const Operator* op,
                                           BuiltinDataAllocator* allocator,
                                           void** builtin_data);

TfLiteStatus ParseStablehloGather(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data);

TfLiteStatus ParseStablehloReduceWindow(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data);

TfLiteStatus ParseStablehloPad(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data);

TfLiteStatus ParseStablehloComposite(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data);

TfLiteStatus ParseStablehloShiftLeft(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data);

TfLiteStatus ParseStablehloCase(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data);

}  // namespace micro

using TfLiteBridgeBuiltinDataAllocator = micro::BuiltinDataAllocator;
using TfLiteBridgeBuiltinParseFunction =
    micro::TfLiteBridgeBuiltinParseFunction;
using micro::CallBuiltinParseFunction;
using micro::ConvertTensorType;
using micro::ParseAbs;
using micro::ParseAdd;
using micro::ParseAddN;
using micro::ParseArgMax;
using micro::ParseArgMin;
using micro::ParseAssignVariable;
using micro::ParseBatchMatMul;
using micro::ParseBatchToSpaceNd;
using micro::ParseBitwiseXor;
using micro::ParseBroadcastArgs;
using micro::ParseBroadcastTo;
using micro::ParseCallOnce;
using micro::ParseCast;
using micro::ParseCeil;
using micro::ParseConcatenation;
using micro::ParseConv2D;
using micro::ParseCos;
using micro::ParseCumsum;
using micro::ParseDepthToSpace;
using micro::ParseDepthwiseConv2D;
using micro::ParseDequantize;
using micro::ParseDiv;
using micro::ParseDynamicUpdateSlice;
using micro::ParseElu;
using micro::ParseEmbeddingLookup;
using micro::ParseEqual;
using micro::ParseExp;
using micro::ParseExpandDims;
using micro::ParseFill;
using micro::ParseFloor;
using micro::ParseFloorDiv;
using micro::ParseFloorMod;
using micro::ParseFullyConnected;
using micro::ParseGather;
using micro::ParseGatherNd;
using micro::ParseGreater;
using micro::ParseGreaterEqual;
using micro::ParseHardSwish;
using micro::ParseIf;
using micro::ParseL2Normalization;
using micro::ParseLeakyRelu;
using micro::ParseLess;
using micro::ParseLessEqual;
using micro::ParseLog;
using micro::ParseLogicalAnd;
using micro::ParseLogicalNot;
using micro::ParseLogicalOr;
using micro::ParseLogistic;
using micro::ParseLogSoftmax;
using micro::ParseLSTM;
using micro::ParseMaximum;
using micro::ParseMinimum;
using micro::ParseMirrorPad;
using micro::ParseMul;
using micro::ParseNeg;
using micro::ParseNotEqual;
using micro::ParsePack;
using micro::ParsePad;
using micro::ParsePadV2;
using micro::ParsePool;
using micro::ParsePow;
using micro::ParsePrelu;
using micro::ParseQuantize;
using micro::ParseReadVariable;
using micro::ParseReducer;
using micro::ParseRelu;
using micro::ParseRelu6;
using micro::ParseReshape;
using micro::ParseResizeBilinear;
using micro::ParseResizeNearestNeighbor;
using micro::ParseReverseV2;
using micro::ParseRightShift;
using micro::ParseRound;
using micro::ParseRsqrt;
using micro::ParseSelect;
using micro::ParseSelectV2;
using micro::ParseShape;
using micro::ParseSin;
using micro::ParseSlice;
using micro::ParseSoftmax;
using micro::ParseSpaceToBatchNd;
using micro::ParseSpaceToDepth;
using micro::ParseSplit;
using micro::ParseSplitV;
using micro::ParseSqrt;
using micro::ParseSquare;
using micro::ParseSquaredDifference;
using micro::ParseSqueeze;
using micro::ParseStablehloCase;
using micro::ParseStablehloComposite;
using micro::ParseStablehloGather;
using micro::ParseStablehloPad;
using micro::ParseStablehloReduceWindow;
using micro::ParseStablehloRngBitGenerator;
using micro::ParseStablehloScatter;
using micro::ParseStablehloShiftLeft;
using micro::ParseStridedSlice;
using micro::ParseSub;
using micro::ParseSvdf;
using micro::ParseTanh;
using micro::ParseTranspose;
using micro::ParseTransposeConv;
using micro::ParseUnidirectionalSequenceLSTM;
using micro::ParseUnpack;
using micro::ParseVarHandle;
using micro::ParseWhile;
using micro::ParseZerosLike;

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_FLATBUFFER_CONVERSIONS_H_
