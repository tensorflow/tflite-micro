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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_elementwise.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_op_utils.h"

extern "C" {
#include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace activations {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct LUTArguments {
  uint8_t* Y;
  const uint8_t* X;
  const uint8_t* LUT;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

using LUTThreadData = ElementwiseThreadData<LUTArguments>;

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void lut_thread_worker(void* context) {
  auto* td = static_cast<LUTThreadData*>(context);
  auto* args = td->args;
  lookup8(args->Y, args->X, args->LUT, td->start, td->element_count);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct LUTOpData : MultiThreadedOpData<LUTArguments, LUTThreadData> {
  // TODO: remove this when better external memory handling is implemented
  // for loading from external mem
  int lut_scratch_idx = -1;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<LUTOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op_data->stack_size, lut_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->threads.size(),
      &op_data->stack_scratch_index));

  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->lut_scratch_idx));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<LUTOpData*>(node->user_data);

  op_data->args.X = tflite::micro::GetTensorData<uint8_t>(
      tflite::micro::GetEvalInput(context, node, 0));
  TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
      context, op_data->args.LUT, tflite::micro::GetEvalInput(context, node, 1),
      op_data->lut_scratch_idx));
  op_data->args.Y = tflite::micro::GetTensorData<uint8_t>(
      tflite::micro::GetEvalOutput(context, node, 0));

  // initialize the dispatcher
  auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  dispatcher->Initialize(lut_thread_worker, op_data->threads.size(),
                         op_data->stack_size, stack);

  void* dispatcher_args[op_data->threads.size()];
  for (int i = 0; i < op_data->threads.size(); i++) {
    dispatcher_args[i] = &op_data->threads[i];
  }

  dispatcher->Invoke(dispatcher_args, op_data->threads.size());

  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_Lookup_8() {
  static TfLiteRegistration r = {ElementwiseInit<activations::LUTOpData>,
                                 nullptr, activations::Prepare,
                                 activations::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
