/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

  TfLiteStatus AddAbs() {
    return AddBuiltin(BuiltinOperator_ABS, Register_ABS(), ParseAbs);
  }

  TfLiteStatus AddAdd(const TFLMRegistration& registration = Register_ADD()) {
    return AddBuiltin(BuiltinOperator_ADD, registration, ParseAdd);
  }

  TfLiteStatus AddAddN() {
    return AddBuiltin(BuiltinOperator_ADD_N, tflite::Register_ADD_N(),
                      ParseAddN);
  }

  TfLiteStatus AddArgMax() {
    return AddBuiltin(BuiltinOperator_ARG_MAX, Register_ARG_MAX(), ParseArgMax);
  }

  TfLiteStatus AddArgMin() {
    return AddBuiltin(BuiltinOperator_ARG_MIN, Register_ARG_MIN(), ParseArgMin);
  }

  TfLiteStatus AddAssignVariable() {
    return AddBuiltin(BuiltinOperator_ASSIGN_VARIABLE,
                      tflite::Register_ASSIGN_VARIABLE(), ParseAssignVariable);
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

  TfLiteStatus AddBatchToSpaceNd() {
    return AddBuiltin(BuiltinOperator_BATCH_TO_SPACE_ND,
                      Register_BATCH_TO_SPACE_ND(), ParseBatchToSpaceNd);
  }

  TfLiteStatus AddBroadcastArgs() {
    return AddBuiltin(BuiltinOperator_BROADCAST_ARGS, Register_BROADCAST_ARGS(),
                      ParseBroadcastArgs);
  }

  TfLiteStatus AddBroadcastTo() {
    return AddBuiltin(BuiltinOperator_BROADCAST_TO, Register_BROADCAST_TO(),
                      ParseBroadcastTo);
  }

  TfLiteStatus AddCallOnce() {
    return AddBuiltin(BuiltinOperator_CALL_ONCE, Register_CALL_ONCE(),
                      ParseCallOnce);
  }

  TfLiteStatus AddCast() {
    return AddBuiltin(BuiltinOperator_CAST, Register_CAST(), ParseCast);
  }

  TfLiteStatus AddCeil() {
    return AddBuiltin(BuiltinOperator_CEIL, Register_CEIL(), ParseCeil);
  }

  TfLiteStatus AddCircularBuffer() {
    return AddCustom("CIRCULAR_BUFFER", tflite::Register_CIRCULAR_BUFFER());
  }

  TfLiteStatus AddConcatenation() {
    return AddBuiltin(BuiltinOperator_CONCATENATION, Register_CONCATENATION(),
                      ParseConcatenation);
  }

  TfLiteStatus AddConv2D(
      const TFLMRegistration& registration = Register_CONV_2D()) {
    return AddBuiltin(BuiltinOperator_CONV_2D, registration, ParseConv2D);
  }

  TfLiteStatus AddCos() {
    return AddBuiltin(BuiltinOperator_COS, tflite::Register_COS(), ParseCos);
  }

  TfLiteStatus AddCumSum() {
    return AddBuiltin(BuiltinOperator_CUMSUM, tflite::Register_CUMSUM(),
                      ParseCumsum);
  }

  TfLiteStatus AddDelay() {
    // TODO(b/286250473): change back name to "Delay" and remove namespace
    return AddCustom("SignalDelay", tflite::tflm_signal::Register_DELAY());
  }

  TfLiteStatus AddDepthToSpace() {
    return AddBuiltin(BuiltinOperator_DEPTH_TO_SPACE,
                      tflite::Register_DEPTH_TO_SPACE(), ParseDepthToSpace);
  }

  TfLiteStatus AddDepthwiseConv2D(
      const TFLMRegistration& registration = Register_DEPTHWISE_CONV_2D()) {
    return AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, registration,
                      ParseDepthwiseConv2D);
  }

  TfLiteStatus AddDequantize() {
    return AddBuiltin(BuiltinOperator_DEQUANTIZE, tflite::Register_DEQUANTIZE(),
                      ParseDequantize);
  }

  TfLiteStatus AddDetectionPostprocess() {
    return AddCustom("TFLite_Detection_PostProcess",
                     tflite::Register_DETECTION_POSTPROCESS());
  }

  TfLiteStatus AddDiv() {
    return AddBuiltin(BuiltinOperator_DIV, tflite::Register_DIV(), ParseDiv);
  }

  TfLiteStatus AddEmbeddingLookup() {
    return AddBuiltin(BuiltinOperator_EMBEDDING_LOOKUP,
                      Register_EMBEDDING_LOOKUP(), ParseEmbeddingLookup);
  }

  TfLiteStatus AddEnergy() {
    // TODO(b/286250473): change back name to "Energy" and remove namespace
    return AddCustom("SignalEnergy", tflite::tflm_signal::Register_ENERGY());
  }

  TfLiteStatus AddElu() {
    return AddBuiltin(BuiltinOperator_ELU, tflite::Register_ELU(), ParseElu);
  }

  TfLiteStatus AddEqual() {
    return AddBuiltin(BuiltinOperator_EQUAL, Register_EQUAL(), ParseEqual);
  }

  TfLiteStatus AddEthosU() {
    TFLMRegistration* registration = tflite::Register_ETHOSU();
    if (registration) {
      return AddCustom(tflite::GetString_ETHOSU(), registration);
    }
    return kTfLiteOk;
  }

  TfLiteStatus AddExp() {
    return AddBuiltin(BuiltinOperator_EXP, Register_EXP(), ParseExp);
  }

  TfLiteStatus AddExpandDims() {
    return AddBuiltin(BuiltinOperator_EXPAND_DIMS, Register_EXPAND_DIMS(),
                      ParseExpandDims);
  }

  TfLiteStatus AddFftAutoScale() {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalFftAutoScale",
                     tflite::tflm_signal::Register_FFT_AUTO_SCALE());
  }

  TfLiteStatus AddFill() {
    return AddBuiltin(BuiltinOperator_FILL, tflite::Register_FILL(), ParseFill);
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

  TfLiteStatus AddFloor() {
    return AddBuiltin(BuiltinOperator_FLOOR, Register_FLOOR(), ParseFloor);
  }

  TfLiteStatus AddFloorDiv() {
    return AddBuiltin(BuiltinOperator_FLOOR_DIV, tflite::Register_FLOOR_DIV(),
                      ParseFloorDiv);
  }

  TfLiteStatus AddFloorMod() {
    return AddBuiltin(BuiltinOperator_FLOOR_MOD, tflite::Register_FLOOR_MOD(),
                      ParseFloorMod);
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

  TfLiteStatus AddGather() {
    return AddBuiltin(BuiltinOperator_GATHER, tflite::Register_GATHER(),
                      ParseGather);
  }

  TfLiteStatus AddGatherNd() {
    return AddBuiltin(BuiltinOperator_GATHER_ND, tflite::Register_GATHER_ND(),
                      ParseGatherNd);
  }

  TfLiteStatus AddGreater() {
    return AddBuiltin(BuiltinOperator_GREATER, Register_GREATER(),
                      ParseGreater);
  }

  TfLiteStatus AddGreaterEqual() {
    return AddBuiltin(BuiltinOperator_GREATER_EQUAL, Register_GREATER_EQUAL(),
                      ParseGreaterEqual);
  }

  TfLiteStatus AddHardSwish() {
    return AddBuiltin(BuiltinOperator_HARD_SWISH, tflite::Register_HARD_SWISH(),
                      ParseHardSwish);
  }

  TfLiteStatus AddIf() {
    return AddBuiltin(BuiltinOperator_IF, tflite::Register_IF(), ParseIf);
  }

  TfLiteStatus AddIrfft(const TFLMRegistration* registration =
                            tflite::tflm_signal::Register_IRFFT()) {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalIrfft", registration);
  }

  TfLiteStatus AddL2Normalization() {
    return AddBuiltin(BuiltinOperator_L2_NORMALIZATION,
                      Register_L2_NORMALIZATION(), ParseL2Normalization);
  }

  TfLiteStatus AddL2Pool2D() {
    return AddBuiltin(BuiltinOperator_L2_POOL_2D, tflite::Register_L2_POOL_2D(),
                      ParsePool);
  }

  TfLiteStatus AddLeakyRelu() {
    return AddBuiltin(BuiltinOperator_LEAKY_RELU, tflite::Register_LEAKY_RELU(),
                      ParseLeakyRelu);
  }

  TfLiteStatus AddLess() {
    return AddBuiltin(BuiltinOperator_LESS, Register_LESS(), ParseLess);
  }

  TfLiteStatus AddLessEqual() {
    return AddBuiltin(BuiltinOperator_LESS_EQUAL, Register_LESS_EQUAL(),
                      ParseLessEqual);
  }

  TfLiteStatus AddLog() {
    return AddBuiltin(BuiltinOperator_LOG, Register_LOG(), ParseLog);
  }

  TfLiteStatus AddLogicalAnd() {
    return AddBuiltin(BuiltinOperator_LOGICAL_AND,
                      tflite::Register_LOGICAL_AND(), ParseLogicalAnd);
  }

  TfLiteStatus AddLogicalNot() {
    return AddBuiltin(BuiltinOperator_LOGICAL_NOT, Register_LOGICAL_NOT(),
                      ParseLogicalNot);
  }

  TfLiteStatus AddLogicalOr() {
    return AddBuiltin(BuiltinOperator_LOGICAL_OR, tflite::Register_LOGICAL_OR(),
                      ParseLogicalOr);
  }

  TfLiteStatus AddLogistic() {
    return AddBuiltin(BuiltinOperator_LOGISTIC, tflite::Register_LOGISTIC(),
                      ParseLogistic);
  }

  TfLiteStatus AddLogSoftmax() {
    return AddBuiltin(BuiltinOperator_LOG_SOFTMAX,
                      tflite::Register_LOG_SOFTMAX(), ParseLogSoftmax);
  }

  TfLiteStatus AddMaximum(
      const TFLMRegistration& registration = Register_MAXIMUM()) {
    return AddBuiltin(BuiltinOperator_MAXIMUM, registration, ParseMaximum);
  }

  TfLiteStatus AddMaxPool2D(
      const TFLMRegistration& registration = Register_MAX_POOL_2D()) {
    return AddBuiltin(BuiltinOperator_MAX_POOL_2D, registration, ParsePool);
  }

  TfLiteStatus AddMirrorPad() {
    return AddBuiltin(BuiltinOperator_MIRROR_PAD, tflite::Register_MIRROR_PAD(),
                      ParseMirrorPad);
  }

  TfLiteStatus AddMean() {
    return AddBuiltin(BuiltinOperator_MEAN, Register_MEAN(), ParseReducer);
  }

  TfLiteStatus AddMinimum(
      const TFLMRegistration& registration = Register_MINIMUM()) {
    return AddBuiltin(BuiltinOperator_MINIMUM, registration, ParseMinimum);
  }

  TfLiteStatus AddMul(const TFLMRegistration& registration = Register_MUL()) {
    return AddBuiltin(BuiltinOperator_MUL, registration, ParseMul);
  }

  TfLiteStatus AddNeg() {
    return AddBuiltin(BuiltinOperator_NEG, Register_NEG(), ParseNeg);
  }

  TfLiteStatus AddNotEqual() {
    return AddBuiltin(BuiltinOperator_NOT_EQUAL, Register_NOT_EQUAL(),
                      ParseNotEqual);
  }

  TfLiteStatus AddOverlapAdd() {
    // TODO(b/286250473): change back name to "OverlapAdd" and remove
    // namespace
    return AddCustom("SignalOverlapAdd",
                     tflite::tflm_signal::Register_OVERLAP_ADD());
  }

  TfLiteStatus AddPack() {
    return AddBuiltin(BuiltinOperator_PACK, Register_PACK(), ParsePack);
  }

  TfLiteStatus AddPad(const TFLMRegistration& registration = Register_PAD()) {
    return AddBuiltin(BuiltinOperator_PAD, registration, ParsePad);
  }

  TfLiteStatus AddPadV2() {
    return AddBuiltin(BuiltinOperator_PADV2, Register_PADV2(), ParsePadV2);
  }

  TfLiteStatus AddPCAN() {
    // TODO(b/286250473): change back name to "PCAN" and remove namespace
    return AddCustom("SignalPCAN", tflite::tflm_signal::Register_PCAN());
  }

  TfLiteStatus AddPrelu() {
    return AddBuiltin(BuiltinOperator_PRELU, tflite::Register_PRELU(),
                      ParsePrelu);
  }

  TfLiteStatus AddQuantize() {
    return AddBuiltin(BuiltinOperator_QUANTIZE, Register_QUANTIZE(),
                      ParseQuantize);
  }

  TfLiteStatus AddReadVariable() {
    return AddBuiltin(BuiltinOperator_READ_VARIABLE,
                      tflite::Register_READ_VARIABLE(), ParseReadVariable);
  }

  TfLiteStatus AddReduceMax() {
    return AddBuiltin(BuiltinOperator_REDUCE_MAX, Register_REDUCE_MAX(),
                      ParseReducer);
  }

  TfLiteStatus AddRelu() {
    return AddBuiltin(BuiltinOperator_RELU, tflite::Register_RELU(), ParseRelu);
  }

  TfLiteStatus AddRelu6() {
    return AddBuiltin(BuiltinOperator_RELU6, tflite::Register_RELU6(),
                      ParseRelu6);
  }

  TfLiteStatus AddReshape() {
    return AddBuiltin(BuiltinOperator_RESHAPE, Register_RESHAPE(),
                      ParseReshape);
  }

  TfLiteStatus AddResizeBilinear() {
    return AddBuiltin(BuiltinOperator_RESIZE_BILINEAR,
                      Register_RESIZE_BILINEAR(), ParseResizeBilinear);
  }

  TfLiteStatus AddResizeNearestNeighbor() {
    return AddBuiltin(BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                      Register_RESIZE_NEAREST_NEIGHBOR(),
                      ParseResizeNearestNeighbor);
  }

  TfLiteStatus AddRfft(const TFLMRegistration* registration =
                           tflite::tflm_signal::Register_RFFT()) {
    // TODO(b/286250473): change back name and remove namespace
    return AddCustom("SignalRfft", registration);
  }

  TfLiteStatus AddRound() {
    return AddBuiltin(BuiltinOperator_ROUND, Register_ROUND(), ParseRound);
  }

  TfLiteStatus AddRsqrt() {
    return AddBuiltin(BuiltinOperator_RSQRT, Register_RSQRT(), ParseRsqrt);
  }

  TfLiteStatus AddSelectV2() {
    return AddBuiltin(BuiltinOperator_SELECT_V2, Register_SELECT_V2(),
                      ParseSelectV2);
  }

  TfLiteStatus AddShape() {
    return AddBuiltin(BuiltinOperator_SHAPE, Register_SHAPE(), ParseShape);
  }

  TfLiteStatus AddSin() {
    return AddBuiltin(BuiltinOperator_SIN, Register_SIN(), ParseSin);
  }

  TfLiteStatus AddSlice() {
    return AddBuiltin(BuiltinOperator_SLICE, Register_SLICE(), ParseSlice);
  }

  TfLiteStatus AddSoftmax(
      const TFLMRegistration& registration = Register_SOFTMAX()) {
    return AddBuiltin(BuiltinOperator_SOFTMAX, registration, ParseSoftmax);
  }

  TfLiteStatus AddSpaceToBatchNd() {
    return AddBuiltin(BuiltinOperator_SPACE_TO_BATCH_ND,
                      Register_SPACE_TO_BATCH_ND(), ParseSpaceToBatchNd);
  }

  TfLiteStatus AddSpaceToDepth() {
    return AddBuiltin(BuiltinOperator_SPACE_TO_DEPTH, Register_SPACE_TO_DEPTH(),
                      ParseSpaceToDepth);
  }

  TfLiteStatus AddSplit() {
    return AddBuiltin(BuiltinOperator_SPLIT, Register_SPLIT(), ParseSplit);
  }

  TfLiteStatus AddSplitV() {
    return AddBuiltin(BuiltinOperator_SPLIT_V, Register_SPLIT_V(), ParseSplitV);
  }

  TfLiteStatus AddSqueeze() {
    return AddBuiltin(BuiltinOperator_SQUEEZE, Register_SQUEEZE(),
                      ParseSqueeze);
  }

  TfLiteStatus AddSqrt() {
    return AddBuiltin(BuiltinOperator_SQRT, Register_SQRT(), ParseSqrt);
  }

  TfLiteStatus AddSquare() {
    return AddBuiltin(BuiltinOperator_SQUARE, Register_SQUARE(), ParseSquare);
  }

  TfLiteStatus AddSquaredDifference() {
    return AddBuiltin(BuiltinOperator_SQUARED_DIFFERENCE,
                      tflite::Register_SQUARED_DIFFERENCE(),
                      ParseSquaredDifference);
  }

  TfLiteStatus AddStridedSlice() {
    return AddBuiltin(BuiltinOperator_STRIDED_SLICE, Register_STRIDED_SLICE(),
                      ParseStridedSlice);
  }

  TfLiteStatus AddStacker() {
    // TODO(b/286250473): change back name to "Stacker" and remove namespace
    return AddCustom("SignalStacker", tflite::tflm_signal::Register_STACKER());
  }

  TfLiteStatus AddSub() {
    return AddBuiltin(BuiltinOperator_SUB, tflite::Register_SUB(), ParseSub);
  }

  TfLiteStatus AddSum() {
    return AddBuiltin(BuiltinOperator_SUM, Register_SUM(), ParseReducer);
  }

  TfLiteStatus AddSvdf(const TFLMRegistration& registration = Register_SVDF()) {
    return AddBuiltin(BuiltinOperator_SVDF, registration, ParseSvdf);
  }

  TfLiteStatus AddTanh() {
    return AddBuiltin(BuiltinOperator_TANH, Register_TANH(), ParseTanh);
  }

  TfLiteStatus AddTransposeConv(
      const TFLMRegistration& registration = Register_TRANSPOSE_CONV()) {
    return AddBuiltin(BuiltinOperator_TRANSPOSE_CONV, registration,
                      ParseTransposeConv);
  }

  TfLiteStatus AddTranspose() {
    return AddBuiltin(BuiltinOperator_TRANSPOSE, Register_TRANSPOSE(),
                      ParseTranspose);
  }

  TfLiteStatus AddUnpack() {
    return AddBuiltin(BuiltinOperator_UNPACK, Register_UNPACK(), ParseUnpack);
  }

  TfLiteStatus AddUnidirectionalSequenceLSTM(
      const TFLMRegistration& registration =
          Register_UNIDIRECTIONAL_SEQUENCE_LSTM()) {
    return AddBuiltin(BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM,
                      registration, ParseUnidirectionalSequenceLSTM);
  }

  TfLiteStatus AddVarHandle() {
    return AddBuiltin(BuiltinOperator_VAR_HANDLE, Register_VAR_HANDLE(),
                      ParseVarHandle);
  }

  TfLiteStatus AddWhile() {
    return AddBuiltin(BuiltinOperator_WHILE, Register_WHILE(), ParseWhile);
  }

  TfLiteStatus AddWindow() {
    // TODO(b/286250473): change back name to "Window" and remove namespace
    return AddCustom("SignalWindow", tflite::tflm_signal::Register_WINDOW());
  }

  TfLiteStatus AddZerosLike() {
    return AddBuiltin(BuiltinOperator_ZEROS_LIKE, Register_ZEROS_LIKE(),
                      ParseZerosLike);
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
