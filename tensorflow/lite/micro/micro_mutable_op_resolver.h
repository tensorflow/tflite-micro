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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_

#include <cstdio>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/kernels/add.h"
#include "tensorflow/lite/micro/kernels/batch_matmul.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/ethosu.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/maximum_minimum.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/mul.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/svdf.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/kernels/unidirectional_sequence_lstm.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
TFLMRegistration* Register_DETECTION_POSTPROCESS();

template <unsigned int tOpCount>
class MicroMutableOpResolver : public MicroOpResolver {
 public:
  TF_LITE_REMOVE_VIRTUAL_DELETE

  explicit MicroMutableOpResolver() {}

  const TFLMRegistration* FindOp(tflite::BuiltinOperator op) const override {
    if (op == BuiltinOperator_CUSTOM) return nullptr;

    for (unsigned int i = 0; i < registrations_len_; ++i) {
      const TFLMRegistration& registration = registrations_[i];
      if (registration.builtin_code == op) {
        return &registration;
      }
    }
    return nullptr;
  }

  const TFLMRegistration* FindOp(const char* op) const override {
    for (unsigned int i = 0; i < registrations_len_; ++i) {
      const TFLMRegistration& registration = registrations_[i];
      if ((registration.builtin_code == BuiltinOperator_CUSTOM) &&
          (strcmp(registration.custom_name, op) == 0)) {
        return &registration;
      }
    }
    return nullptr;
  }

  TfLiteBridgeBuiltinParseFunction GetOpDataParser(
      BuiltinOperator op) const override {
    TFLITE_DCHECK(num_buitin_ops_ <= tOpCount);
    for (unsigned int i = 0; i < num_buitin_ops_; ++i) {
      if (builtin_codes_[i] == op) return builtin_parsers_[i];
    }
    return nullptr;
  }

  // Registers a Custom Operator with the MicroOpResolver.
  //
  // Only the first call for a given name will be successful. i.e. if this
  // function is called again for a previously added Custom Operator, the
  // MicroOpResolver will be unchanged and this function will return
  // kTfLiteError.
  TfLiteStatus AddCustom(const char* name,
                         const TFLMRegistration* registration) {
    if (registrations_len_ >= tOpCount) {
      MicroPrintf(
          "Couldn't register custom op '%s', resolver size is too"
          "small (%d)",
          name, tOpCount);
      return kTfLiteError;
    }

    if (FindOp(name) != nullptr) {
      MicroPrintf("Calling AddCustom for the same op more than once ");
      MicroPrintf("is not supported (Op: %s).", name);
      return kTfLiteError;
    }

    TFLMRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = BuiltinOperator_CUSTOM;
    new_registration->custom_name = name;
    return kTfLiteOk;
  }

  // The Add* functions below add the various Builtin operators to the
  // MicroMutableOpResolver object.

  TfLiteStatus AddAbs(const TFLMRegistration& registration = Register_ABS()) {
    return AddBuiltin(BuiltinOperator_ABS, registration, ParseAbs);
  }

  TfLiteStatus AddAdd(const TFLMRegistration& registration = Register_ADD()) {
    return AddBuiltin(BuiltinOperator_ADD, registration, ParseAdd);
  }

  TfLiteStatus AddAddN(
      const TFLMRegistration& registration = Register_ADD_N()) {
    return AddBuiltin(BuiltinOperator_ADD_N, registration, ParseAddN);
  }

  TfLiteStatus AddArgMax(
      const TFLMRegistration& registration = Register_ARG_MAX()) {
    return AddBuiltin(BuiltinOperator_ARG_MAX, registration, ParseArgMax);
  }

  TfLiteStatus AddArgMin(
      const TFLMRegistration& registration = Register_ARG_MIN()) {
    return AddBuiltin(BuiltinOperator_ARG_MIN, registration, ParseArgMin);
  }

  TfLiteStatus AddAssignVariable(const TFLMRegistration& registration =
                                     tflite::Register_ASSIGN_VARIABLE()) {
    return AddBuiltin(BuiltinOperator_ASSIGN_VARIABLE, registration,
                      ParseAssignVariable);
  }

  TfLiteStatus AddAveragePool2D(
      const TFLMRegistration& registration = Register_AVERAGE_POOL_2D()) {
    return AddBuiltin(BuiltinOperator_AVERAGE_POOL_2D, registration, ParsePool);
  }

  TfLiteStatus AddBatchMatMul(
      const TFLMRegistration& registration = Register_BATCH_MATMUL()) {
    return AddBuiltin(BuiltinOperator_BATCH_MATMUL, registration,
                      ParseBatchMatMul);
  }

  TfLiteStatus AddBatchToSpaceNd(
      const TFLMRegistration& registration = Register_BATCH_TO_SPACE_ND()) {
    return AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND, registration,
                      ParseBatchToSpaceNd);
  }

  TfLiteStatus AddBroadcastArgs(
      const TFLMRegistration& registration = Register_BROADCAST_ARGS()) {
    return AddBuiltin(BuiltinOperator_BROADCAST_ARGS, registration,
                      ParseBroadcastArgs);
  }

  TfLiteStatus AddBroadcastTo(
      const TFLMRegistration& registration = Register_BROADCAST_TO()) {
    return AddBuiltin(BuiltinOperator_BROADCAST_TO, registration,
                      ParseBroadcastTo);
  }

  TfLiteStatus AddCallOnce(
      const TFLMRegistration& registration = Register_CALL_ONCE()) {
    return AddBuiltin(BuiltinOperator_CALL_ONCE, registration, ParseCallOnce);
  }

  TfLiteStatus AddCast(const TFLMRegistration& registration = Register_CAST()) {
    return AddBuiltin(BuiltinOperator_CAST, registration, ParseCast);
  }

  TfLiteStatus AddCeil(const TFLMRegistration& registration = Register_CEIL()) {
    return AddBuiltin(BuiltinOperator_CEIL, registration, ParseCeil);
  }

  TfLiteStatus AddCircularBuffer() {
    return AddCustom("CIRCULAR_BUFFER", tflite::Register_CIRCULAR_BUFFER());
  }

  TfLiteStatus AddConcatenation(
      const TFLMRegistration& registration = Register_CONCATENATION()) {
    return AddBuiltin(BuiltinOperator_CONCATENATION, registration,
                      ParseConcatenation);
  }

  TfLiteStatus AddConv2D(
      const TFLMRegistration& registration = Register_CONV_2D()) {
    return AddBuiltin(BuiltinOperator_CONV_2D, registration, ParseConv2D);
  }

  TfLiteStatus AddCos(const TFLMRegistration& registration = Register_COS()) {
    return AddBuiltin(BuiltinOperator_COS, registration, ParseCos);
  }

  TfLiteStatus AddCumSum(
      const TFLMRegistration& registration = Register_CUMSUM()) {
    return AddBuiltin(BuiltinOperator_CUMSUM, registration, ParseCumsum);
  }

  TfLiteStatus AddDelay() {
    // TODO(b/286250473): change back name to "Delay" and remove namespace
    return AddCustom("SignalDelay", tflite::tflm_signal::Register_DELAY());
  }

  TfLiteStatus AddDepthToSpace(const TFLMRegistration& registration =
                                   tflite::Register_DEPTH_TO_SPACE()) {
    return AddBuiltin(BuiltinOperator_DEPTH_TO_SPACE, registration,
                      ParseDepthToSpace);
  }

  TfLiteStatus AddDepthwiseConv2D(
      const TFLMRegistration& registration = Register_DEPTHWISE_CONV_2D()) {
    return AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, registration,
                      ParseDepthwiseConv2D);
  }

  TfLiteStatus AddDequantize(
      const TFLMRegistration& registration = Register_DEQUANTIZE()) {
    return AddBuiltin(BuiltinOperator_DEQUANTIZE, registration,
                      ParseDequantize);
  }

  TfLiteStatus AddDetectionPostprocess() {
    return AddCustom("TFLite_Detection_PostProcess",
                     tflite::Register_DETECTION_POSTPROCESS());
  }

  TfLiteStatus AddDiv(
      const TFLMRegistration& registration = tflite::Register_DIV()) {
    return AddBuiltin(BuiltinOperator_DIV, registration, ParseDiv);
  }

  TfLiteStatus AddEmbeddingLookup(
      const TFLMRegistration& registration = Register_EMBEDDING_LOOKUP()) {
    return AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP, registration,
                      ParseEmbeddingLookup);
  }

  TfLiteStatus AddEnergy() {
    // TODO(b/286250473): change back name to "Energy" and remove namespace
    return AddCustom("SignalEnergy", tflite::tflm_signal::Register_ENERGY());
  }

  TfLiteStatus AddElu(
      const TFLMRegistration& registration = tflite::Register_ELU()) {
    return AddBuiltin(BuiltinOperator_ELU, registration, ParseElu);
  }

  TfLiteStatus AddEqual(
      const TFLMRegistration& registration = Register_EQUAL()) {
    return AddBuiltin(BuiltinOperator_EQUAL, registration, ParseEqual);
  }

  TfLiteStatus AddEthosU() {
    TFLMRegistration* registration = tflite::Register_ETHOSU();
    if (registration) {
      return AddCustom(tflite::GetString_ETHOSU(), registration);
    }
    return kTfLiteOk;
  }

  TfLiteStatus AddExp(const TFLMRegistration& registration = Register_EXP()) {
    return AddBuiltin(BuiltinOperator_EXP, registration, ParseExp);
  }

  TfLiteStatus AddExpandDims(
      const TFLMRegistration& registration = Register_EXPAND_DIMS()) {
    return AddBuiltin(BuiltinOperator_EXPAND_DIMS, registration,
                      ParseExpandDims);
  }

  TfLiteStatus AddFftAutoScale() {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalFftAutoScale",
                     tflite::tflm_signal::Register_FFT_AUTO_SCALE());
  }

  TfLiteStatus AddFill(
      const TFLMRegistration& registration = tflite::Register_FILL()) {
    return AddBuiltin(BuiltinOperator_FILL, registration, ParseFill);
  }

  TfLiteStatus AddFilterBank() {
    // TODO(b/286250473): change back name to "FilterBank" and remove namespace
    return AddCustom("SignalFilterBank",
                     tflite::tflm_signal::Register_FILTER_BANK());
  }
  TfLiteStatus AddFilterBankLog() {
    // TODO(b/286250473): change back name to "FilterBankLog" and remove
    // namespace
    return AddCustom("SignalFilterBankLog",
                     tflite::tflm_signal::Register_FILTER_BANK_LOG());
  }
  TfLiteStatus AddFilterBankSquareRoot() {
    // TODO(b/286250473): change back name to "FilterBankSquareRoot" and remove
    // namespace
    return AddCustom("SignalFilterBankSquareRoot",
                     tflite::tflm_signal::Register_FILTER_BANK_SQUARE_ROOT());
  }
  TfLiteStatus AddFilterBankSpectralSubtraction() {
    // TODO(b/286250473): change back name to "FilterBankSpectralSubtraction"
    // and remove namespace
    return AddCustom(
        "SignalFilterBankSpectralSubtraction",
        tflite::tflm_signal::Register_FILTER_BANK_SPECTRAL_SUBTRACTION());
  }

  TfLiteStatus AddFloor(
      const TFLMRegistration& registration = Register_FLOOR()) {
    return AddBuiltin(BuiltinOperator_FLOOR, registration, ParseFloor);
  }

  TfLiteStatus AddFloorDiv(
      const TFLMRegistration& registration = tflite::Register_FLOOR_DIV()) {
    return AddBuiltin(BuiltinOperator_FLOOR_DIV, registration, ParseFloorDiv);
  }

  TfLiteStatus AddFloorMod(
      const TFLMRegistration& registration = tflite::Register_FLOOR_MOD()) {
    return AddBuiltin(BuiltinOperator_FLOOR_MOD, registration, ParseFloorMod);
  }

  TfLiteStatus AddFramer() {
    // TODO(b/286250473): change back name to "Framer" and remove namespace
    return AddCustom("SignalFramer", tflite::tflm_signal::Register_FRAMER());
  }

  TfLiteStatus AddFullyConnected(
      const TFLMRegistration& registration = Register_FULLY_CONNECTED()) {
    return AddBuiltin(BuiltinOperator_FULLY_CONNECTED, registration,
                      ParseFullyConnected);
  }

  TfLiteStatus AddGather(
      const TFLMRegistration& registration = tflite::Register_GATHER()) {
    return AddBuiltin(BuiltinOperator_GATHER, registration, ParseGather);
  }
  TfLiteStatus AddGatherNd(
      const TFLMRegistration& registration = tflite::Register_GATHER_ND()) {
    return AddBuiltin(BuiltinOperator_GATHER_ND, registration, ParseGatherNd);
  }

  TfLiteStatus AddGreater(
      const TFLMRegistration& registration = Register_GREATER()) {
    return AddBuiltin(BuiltinOperator_GREATER, registration, ParseGreater);
  }

  TfLiteStatus AddGreaterEqual(
      const TFLMRegistration& registration = Register_GREATER_EQUAL()) {
    return AddBuiltin(BuiltinOperator_GREATER_EQUAL, registration,
                      ParseGreaterEqual);
  }

  TfLiteStatus AddHardSwish(
      const TFLMRegistration& registration = tflite::Register_HARD_SWISH()) {
    return AddBuiltin(BuiltinOperator_HARD_SWISH, registration, ParseHardSwish);
  }

  TfLiteStatus AddIf(
      const TFLMRegistration& registration = tflite::Register_IF()) {
    return AddBuiltin(BuiltinOperator_IF, registration, ParseIf);
  }

  TfLiteStatus AddIrfft(const TFLMRegistration* registration =
                            tflite::tflm_signal::Register_IRFFT()) {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalIrfft", registration);
  }

  TfLiteStatus AddL2Normalization(
      const TFLMRegistration& registration = Register_L2_NORMALIZATION()) {
    return AddBuiltin(BuiltinOperator_L2_NORMALIZATION, registration,
                      ParseL2Normalization);
  }

  TfLiteStatus AddL2Pool2D(
      const TFLMRegistration& registration = tflite::Register_L2_POOL_2D()) {
    return AddBuiltin(BuiltinOperator_L2_POOL_2D, registration, ParsePool);
  }

  TfLiteStatus AddLeakyRelu(
      const TFLMRegistration& registration = tflite::Register_LEAKY_RELU()) {
    return AddBuiltin(BuiltinOperator_LEAKY_RELU, registration, ParseLeakyRelu);
  }

  TfLiteStatus AddLess(const TFLMRegistration& registration = Register_LESS()) {
    return AddBuiltin(BuiltinOperator_LESS, registration, ParseLess);
  }

  TfLiteStatus AddLessEqual(
      const TFLMRegistration& registration = Register_LESS_EQUAL()) {
    return AddBuiltin(BuiltinOperator_LESS_EQUAL, registration, ParseLessEqual);
  }

  TfLiteStatus AddLog(const TFLMRegistration& registration = Register_LOG()) {
    return AddBuiltin(BuiltinOperator_LOG, registration, ParseLog);
  }

  TfLiteStatus AddLogicalAnd(
      const TFLMRegistration& registration = tflite::Register_LOGICAL_AND()) {
    return AddBuiltin(BuiltinOperator_LOGICAL_AND, registration,
                      ParseLogicalAnd);
  }

  TfLiteStatus AddLogicalNot(
      const TFLMRegistration& registration = Register_LOGICAL_NOT()) {
    return AddBuiltin(BuiltinOperator_LOGICAL_NOT, registration,
                      ParseLogicalNot);
  }

  TfLiteStatus AddLogicalOr(
      const TFLMRegistration& registration = tflite::Register_LOGICAL_OR()) {
    return AddBuiltin(BuiltinOperator_LOGICAL_OR, registration, ParseLogicalOr);
  }

  TfLiteStatus AddLogistic(
      const TFLMRegistration& registration = tflite::Register_LOGISTIC()) {
    return AddBuiltin(BuiltinOperator_LOGISTIC, registration, ParseLogistic);
  }

  TfLiteStatus AddLogSoftmax(
      const TFLMRegistration& registration = tflite::Register_LOG_SOFTMAX()) {
    return AddBuiltin(BuiltinOperator_LOG_SOFTMAX, registration,
                      ParseLogSoftmax);
  }

  TfLiteStatus AddMaximum(
      const TFLMRegistration& registration = Register_MAXIMUM()) {
    return AddBuiltin(BuiltinOperator_MAXIMUM, registration, ParseMaximum);
  }

  TfLiteStatus AddMaxPool2D(
      const TFLMRegistration& registration = Register_MAX_POOL_2D()) {
    return AddBuiltin(BuiltinOperator_MAX_POOL_2D, registration, ParsePool);
  }

  TfLiteStatus AddMirrorPad(
      const TFLMRegistration& registration = tflite::Register_MIRROR_PAD()) {
    return AddBuiltin(BuiltinOperator_MIRROR_PAD, registration, ParseMirrorPad);
  }

  TfLiteStatus AddMean(const TFLMRegistration& registration = Register_MEAN()) {
    return AddBuiltin(BuiltinOperator_MEAN, registration, ParseReducer);
  }

  TfLiteStatus AddMinimum(
      const TFLMRegistration& registration = Register_MINIMUM()) {
    return AddBuiltin(BuiltinOperator_MINIMUM, registration, ParseMinimum);
  }

  TfLiteStatus AddMul(const TFLMRegistration& registration = Register_MUL()) {
    return AddBuiltin(BuiltinOperator_MUL, registration, ParseMul);
  }

  TfLiteStatus AddNeg(const TFLMRegistration& registration = Register_NEG()) {
    return AddBuiltin(BuiltinOperator_NEG, registration, ParseNeg);
  }

  TfLiteStatus AddNotEqual(
      const TFLMRegistration& registration = Register_NOT_EQUAL()) {
    return AddBuiltin(BuiltinOperator_NOT_EQUAL, registration, ParseNotEqual);
  }

  TfLiteStatus AddOverlapAdd() {
    // TODO(b/286250473): change back name to "OverlapAdd" and remove
    // namespace
    return AddCustom("SignalOverlapAdd",
                     tflite::tflm_signal::Register_OVERLAP_ADD());
  }

  TfLiteStatus AddPack(const TFLMRegistration& registration = Register_PACK()) {
    return AddBuiltin(BuiltinOperator_PACK, registration, ParsePack);
  }

  TfLiteStatus AddPad(const TFLMRegistration& registration = Register_PAD()) {
    return AddBuiltin(BuiltinOperator_PAD, registration, ParsePad);
  }

  TfLiteStatus AddPadV2(
      const TFLMRegistration& registration = Register_PADV2()) {
    return AddBuiltin(BuiltinOperator_PADV2, registration, ParsePadV2);
  }

  TfLiteStatus AddPCAN() {
    // TODO(b/286250473): change back name to "PCAN" and remove namespace
    return AddCustom("SignalPCAN", tflite::tflm_signal::Register_PCAN());
  }

  TfLiteStatus AddPrelu(
      const TFLMRegistration& registration = tflite::Register_PRELU()) {
    return AddBuiltin(BuiltinOperator_PRELU, registration, ParsePrelu);
  }

  TfLiteStatus AddQuantize(
      const TFLMRegistration& registration = Register_QUANTIZE()) {
    return AddBuiltin(BuiltinOperator_QUANTIZE, registration, ParseQuantize);
  }

  TfLiteStatus AddReadVariable(
      const TFLMRegistration& registration = tflite::Register_READ_VARIABLE()) {
    return AddBuiltin(BuiltinOperator_READ_VARIABLE, registration,
                      ParseReadVariable);
  }

  TfLiteStatus AddReduceMax(
      const TFLMRegistration& registration = Register_REDUCE_MAX()) {
    return AddBuiltin(BuiltinOperator_REDUCE_MAX, registration, ParseReducer);
  }

  TfLiteStatus AddReduceMin(
      const TFLMRegistration& registration = Register_REDUCE_MIN()) {
    return AddBuiltin(BuiltinOperator_REDUCE_MIN, registration, ParseReducer);
  }

  TfLiteStatus AddRelu(
      const TFLMRegistration& registration = tflite::Register_RELU()) {
    return AddBuiltin(BuiltinOperator_RELU, registration, ParseRelu);
  }

  TfLiteStatus AddRelu6(
      const TFLMRegistration& registration = tflite::Register_RELU6()) {
    return AddBuiltin(BuiltinOperator_RELU6, registration, ParseRelu6);
  }

  TfLiteStatus AddReshape(
      const TFLMRegistration& registration = Register_RESHAPE()) {
    return AddBuiltin(BuiltinOperator_RESHAPE, registration, ParseReshape);
  }

  TfLiteStatus AddResizeBilinear(
      const TFLMRegistration& registration = Register_RESIZE_BILINEAR()) {
    return AddBuiltin(BuiltinOperator_RESIZE_BILINEAR, registration,
                      ParseResizeBilinear);
  }

  TfLiteStatus AddResizeNearestNeighbor() {
    return AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                      Register_RESIZE_NEAREST_NEIGHBOR(),
                      ParseResizeNearestNeighbor);
  }

  TfLiteStatus AddReverseV2(
      const TFLMRegistration& registration = Register_REVERSE_V2()) {
    return AddBuiltin(BuiltinOperator_REVERSE_V2, registration, ParseReverseV2);
  }

  TfLiteStatus AddRfft(const TFLMRegistration* registration =
                           tflite::tflm_signal::Register_RFFT()) {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalRfft", registration);
  }

  TfLiteStatus AddRound(
      const TFLMRegistration& registration = Register_ROUND()) {
    return AddBuiltin(BuiltinOperator_ROUND, registration, ParseRound);
  }

  TfLiteStatus AddRsqrt(
      const TFLMRegistration& registration = Register_RSQRT()) {
    return AddBuiltin(BuiltinOperator_RSQRT, registration, ParseRsqrt);
  }
  TfLiteStatus AddSelectV2(
      const TFLMRegistration& registration = Register_SELECT_V2()) {
    return AddBuiltin(BuiltinOperator_SELECT_V2, registration, ParseSelectV2);
  }
  TfLiteStatus AddShape(
      const TFLMRegistration& registration = Register_SHAPE()) {
    return AddBuiltin(BuiltinOperator_SHAPE, registration, ParseShape);
  }

  TfLiteStatus AddSin(const TFLMRegistration& registration = Register_SIN()) {
    return AddBuiltin(BuiltinOperator_SIN, registration, ParseSin);
  }

  TfLiteStatus AddSlice(
      const TFLMRegistration& registration = Register_SLICE()) {
    return AddBuiltin(BuiltinOperator_SLICE, registration, ParseSlice);
  }

  TfLiteStatus AddSoftmax(
      const TFLMRegistration& registration = Register_SOFTMAX()) {
    return AddBuiltin(BuiltinOperator_SOFTMAX, registration, ParseSoftmax);
  }

  TfLiteStatus AddSpaceToBatchNd(
      const TFLMRegistration& registration = Register_SPACE_TO_BATCH_ND()) {
    return AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND, registration,
                      ParseSpaceToBatchNd);
  }

  TfLiteStatus AddSpaceToDepth(
      const TFLMRegistration& registration = Register_SPACE_TO_DEPTH()) {
    return AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, registration,
                      ParseSpaceToDepth);
  }

  TfLiteStatus AddSplit(
      const TFLMRegistration& registration = Register_SPLIT()) {
    return AddBuiltin(BuiltinOperator_SPLIT, registration, ParseSplit);
  }

  TfLiteStatus AddSplitV(
      const TFLMRegistration& registration = Register_SPLIT_V()) {
    return AddBuiltin(BuiltinOperator_SPLIT_V, registration, ParseSplitV);
  }

  TfLiteStatus AddSqueeze(
      const TFLMRegistration& registration = Register_SQUEEZE()) {
    return AddBuiltin(BuiltinOperator_SQUEEZE, registration, ParseSqueeze);
  }

  TfLiteStatus AddSqrt(const TFLMRegistration& registration = Register_SQRT()) {
    return AddBuiltin(BuiltinOperator_SQRT, registration, ParseSqrt);
  }

  TfLiteStatus AddSquare(
      const TFLMRegistration& registration = Register_SQUARE()) {
    return AddBuiltin(BuiltinOperator_SQUARE, registration, ParseSquare);
  }

  TfLiteStatus AddSquaredDifference(const TFLMRegistration& registration =
                                        tflite::Register_SQUARED_DIFFERENCE()) {
    return AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE, registration,
                      ParseSquaredDifference);
  }

  TfLiteStatus AddStridedSlice(
      const TFLMRegistration& registration = Register_STRIDED_SLICE()) {
    return AddBuiltin(BuiltinOperator_STRIDED_SLICE, registration,
                      ParseStridedSlice);
  }

  TfLiteStatus AddStacker() {
    // TODO(b/286250473): change back name to "Stacker" and remove namespace
    return AddCustom("SignalStacker", tflite::tflm_signal::Register_STACKER());
  }

  TfLiteStatus AddSub(
      const TFLMRegistration& registration = tflite::Register_SUB()) {
    return AddBuiltin(BuiltinOperator_SUB, registration, ParseSub);
  }

  TfLiteStatus AddSum(const TFLMRegistration& registration = Register_SUM()) {
    return AddBuiltin(BuiltinOperator_SUM, registration, ParseReducer);
  }

  TfLiteStatus AddSvdf(const TFLMRegistration& registration = Register_SVDF()) {
    return AddBuiltin(BuiltinOperator_SVDF, registration, ParseSvdf);
  }

  TfLiteStatus AddTanh(const TFLMRegistration& registration = Register_TANH()) {
    return AddBuiltin(BuiltinOperator_TANH, registration, ParseTanh);
  }

  TfLiteStatus AddTransposeConv(
      const TFLMRegistration& registration = Register_TRANSPOSE_CONV()) {
    return AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, registration,
                      ParseTransposeConv);
  }

  TfLiteStatus AddTranspose(
      const TFLMRegistration& registration = Register_TRANSPOSE()) {
    return AddBuiltin(BuiltinOperator_TRANSPOSE, registration, ParseTranspose);
  }

  TfLiteStatus AddUnpack(
      const TFLMRegistration& registration = Register_UNPACK()) {
    return AddBuiltin(BuiltinOperator_UNPACK, registration, ParseUnpack);
  }

  TfLiteStatus AddUnidirectionalSequenceLSTM(
      const TFLMRegistration& registration =
          Register_UNIDIRECTIONAL_SEQUENCE_LSTM()) {
    return AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
                      registration, ParseUnidirectionalSequenceLSTM);
  }

  TfLiteStatus AddVarHandle(
      const TFLMRegistration& registration = Register_VAR_HANDLE()) {
    return AddBuiltin(BuiltinOperator_VAR_HANDLE, registration, ParseVarHandle);
  }

  TfLiteStatus AddWhile(
      const TFLMRegistration& registration = Register_WHILE()) {
    return AddBuiltin(BuiltinOperator_WHILE, registration, ParseWhile);
  }

  TfLiteStatus AddWindow() {
    // TODO(b/286250473): change back name to "Window" and remove namespace
    return AddCustom("SignalWindow", tflite::tflm_signal::Register_WINDOW());
  }

  TfLiteStatus AddZerosLike(
      const TFLMRegistration& registration = Register_ZEROS_LIKE()) {
    return AddBuiltin(BuiltinOperator_ZEROS_LIKE, registration, ParseZerosLike);
  }

  unsigned int GetRegistrationLength() { return registrations_len_; }

 private:
  TfLiteStatus AddBuiltin(tflite::BuiltinOperator op,
                          const TFLMRegistration& registration,
                          TfLiteBridgeBuiltinParseFunction parser) {
    if (op == BuiltinOperator_CUSTOM) {
      MicroPrintf("Invalid parameter BuiltinOperator_CUSTOM to the ");
      MicroPrintf("AddBuiltin function.");
      return kTfLiteError;
    }

    if (FindOp(op) != nullptr) {
      MicroPrintf("Calling AddBuiltin with the same op more than ");
      MicroPrintf("once is not supported (Op: #%d).", op);
      return kTfLiteError;
    }

    if (registrations_len_ >= tOpCount) {
      MicroPrintf("Couldn't register builtin op #%d, resolver size ", op);
      MicroPrintf("is too small (%d).", tOpCount);
      return kTfLiteError;
    }

    registrations_[registrations_len_] = registration;
    // Strictly speaking, the builtin_code is not necessary for TFLM but
    // filling it in regardless.
    registrations_[registrations_len_].builtin_code = op;
    registrations_len_++;

    builtin_codes_[num_buitin_ops_] = op;
    builtin_parsers_[num_buitin_ops_] = parser;
    num_buitin_ops_++;

    return kTfLiteOk;
  }

  TFLMRegistration registrations_[tOpCount];
  unsigned int registrations_len_ = 0;

  // Arrays (and counter) to store the builtin codes and their corresponding
  // parse functions as these are registered with the Op Resolver.
  BuiltinOperator builtin_codes_[tOpCount];
  TfLiteBridgeBuiltinParseFunction builtin_parsers_[tOpCount];
  unsigned int num_buitin_ops_ = 0;
};

};  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_MUTABLE_OP_RESOLVER_H_
