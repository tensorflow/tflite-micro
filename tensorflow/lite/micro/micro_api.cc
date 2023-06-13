/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/model_runner/output_handler.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_api.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
// clang-format off
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// clang-format on

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

}  // namespace

// The name of this function is important for Arduino compatibility.
int micro_model_setup(const void* model_data, int kTensorArenaSize,
                      uint8_t* tensor_arena) {

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return 1;
  }
  MicroPrintf("Create Interpretor");

  // clang-format off
    // Only Pull in functions that are needed by the model
  static tflite::MicroMutableOpResolver<97> micro_op_resolver;
    micro_op_resolver.AddAddAbs();
    micro_op_resolver.AddAddAdd();
    micro_op_resolver.AddAddAddN();
    micro_op_resolver.AddAddArgMax();
    micro_op_resolver.AddAddArgMin();
    micro_op_resolver.AddAddAssignVariable();
    micro_op_resolver.AddAddAveragePool2D();
    micro_op_resolver.AddAddBatchToSpaceNd();
    micro_op_resolver.AddAddBroadcastArgs();
    micro_op_resolver.AddAddBroadcastTo();
    micro_op_resolver.AddAddCallOnce();
    micro_op_resolver.AddAddCast();
    micro_op_resolver.AddAddCeil();
    micro_op_resolver.AddAddCircularBuffer();
    micro_op_resolver.AddAddConcatenation();
    micro_op_resolver.AddAddConv2D();
    micro_op_resolver.AddAddCos();
    micro_op_resolver.AddAddCumSum();
    micro_op_resolver.AddAddDepthToSpace();
    micro_op_resolver.AddAddDepthwiseConv2D();
    micro_op_resolver.AddAddDequantize();
    micro_op_resolver.AddAddDetectionPostprocess();
    micro_op_resolver.AddAddDiv();
    micro_op_resolver.AddAddElu();
    micro_op_resolver.AddAddEqual();
    micro_op_resolver.AddAddEthosU();
    micro_op_resolver.AddAddExp();
    micro_op_resolver.AddAddExpandDims();
    micro_op_resolver.AddAddFill();
    micro_op_resolver.AddAddFloor();
    micro_op_resolver.AddAddFloorDiv();
    micro_op_resolver.AddAddFloorMod();
    micro_op_resolver.AddAddFullyConnected();
    micro_op_resolver.AddAddGather();
    micro_op_resolver.AddAddGatherNd();
    micro_op_resolver.AddAddGreater();
    micro_op_resolver.AddAddGreaterEqual();
    micro_op_resolver.AddAddHardSwish();
    micro_op_resolver.AddAddIf();
    micro_op_resolver.AddAddL2Normalization();
    micro_op_resolver.AddAddL2Pool2D();
    micro_op_resolver.AddAddLeakyRelu();
    micro_op_resolver.AddAddLess();
    micro_op_resolver.AddAddLessEqual();
    micro_op_resolver.AddAddLog();
    micro_op_resolver.AddAddLogSoftmax();
    micro_op_resolver.AddAddLogicalAnd();
    micro_op_resolver.AddAddLogicalNot();
    micro_op_resolver.AddAddLogicalOr();
    micro_op_resolver.AddAddLogistic();
    micro_op_resolver.AddAddMaxPool2D();
    micro_op_resolver.AddAddMaximum();
    micro_op_resolver.AddAddMean();
    micro_op_resolver.AddAddMinimum();
    micro_op_resolver.AddAddMirrorPad();
    micro_op_resolver.AddAddMul();
    micro_op_resolver.AddAddNeg();
    micro_op_resolver.AddAddNotEqual();
    micro_op_resolver.AddAddPack();
    micro_op_resolver.AddAddPad();
    micro_op_resolver.AddAddPadV2();
    micro_op_resolver.AddAddPrelu();
    micro_op_resolver.AddAddQuantize();
    micro_op_resolver.AddAddReadVariable();
    micro_op_resolver.AddAddReduceMax();
    micro_op_resolver.AddAddRelu();
    micro_op_resolver.AddAddRelu6();
    micro_op_resolver.AddAddReshape();
    micro_op_resolver.AddAddResizeBilinear();
    micro_op_resolver.AddAddResizeNearestNeighbor();
    micro_op_resolver.AddAddRound();
    micro_op_resolver.AddAddRsqrt();
    micro_op_resolver.AddAddSelectV2();
    micro_op_resolver.AddAddShape();
    micro_op_resolver.AddAddSin();
    micro_op_resolver.AddAddSlice();
    micro_op_resolver.AddAddSoftmax();
    micro_op_resolver.AddAddSpaceToBatchNd();
    micro_op_resolver.AddAddSpaceToDepth();
    micro_op_resolver.AddAddSplit();
    micro_op_resolver.AddAddSplitV();
    micro_op_resolver.AddAddSqrt();
    micro_op_resolver.AddAddSquare();
    micro_op_resolver.AddAddSquaredDifference();
    micro_op_resolver.AddAddSqueeze();
    micro_op_resolver.AddAddStridedSlice();
    micro_op_resolver.AddAddSub();
    micro_op_resolver.AddAddSum();
    micro_op_resolver.AddAddSvdf();
    micro_op_resolver.AddAddTanh();
    micro_op_resolver.AddAddTranspose();
    micro_op_resolver.AddAddTransposeConv();
    micro_op_resolver.AddAddUnidirectionalSequenceLSTM();
    micro_op_resolver.AddAddUnpack();
    micro_op_resolver.AddAddVarHandle();
    micro_op_resolver.AddAddWhile();
    micro_op_resolver.AddAddZerosLike();
  // clang-format on

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  MicroPrintf("Allocate Tensor Arena");

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    MicroPrintf("Allocate failed for tensor size %d",
                         kTensorArenaSize);
    return 2;
  }

  MicroPrintf("FOUND TENSOR SIZE: %d",
                       interpreter->arena_used_bytes());

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  return 0;
}


int8_t quantize_input(uint8_t input_value, float scale_factor, int zero_bias){
	int tmp_value = input_value+zero_bias;
	tmp_value*=(float)scale_factor;

  if (tmp_value < -128) {
        tmp_value = -128;
      }
      if (tmp_value > 127) {
        tmp_value = 127;
      }

	return (int8_t)tmp_value;
}

int micro_model_invoke(unsigned char* input_data, int num_inputs, float* results,
                       int num_outputs, float scale_factor, int zero_bias) {


  if (model_input->type == kTfLiteFloat32) {
    for (int i = 0; i < num_inputs; i++) {
      model_input->data.f[i] = (float)input_data[i];
    }
  }

  if (model_input->type == kTfLiteUInt8) {
    for (int i = 0; i < num_inputs; i++) {
      model_input->data.uint8[i] = input_data[i];
    }
  }

  if (model_input->type == kTfLiteInt8) {
    for (int i = 0; i < num_inputs; i++) {
      model_input->data.int8[i] = quantize_input(input_data[i], scale_factor, zero_bias);
    }
  }

  // Run inference, and report any error.
  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on index");
    return 1;
  }

  // Read the predicted y value from the model's output tensor
  if (model_output->type == kTfLiteFloat32) {
  for (int i = 0; i < num_outputs; i++) {
    results[i] = model_output->data.f[i];
  }
  }

  if (model_output->type == kTfLiteUInt8) {
  for (int i = 0; i < num_outputs; i++) {
    results[i] = (float)model_output->data.uint8[i];
  }
  }


  if (model_output->type == kTfLiteInt8) {
  for (int i = 0; i < num_outputs; i++) {
    results[i] = (float)model_output->data.int8[i];
  }
  }


  return 0;
}
