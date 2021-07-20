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
namespace avgpool_global {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct AvgPoolGlobalArguments {
  nn_image_t* Y;
  const nn_image_t* X;

  nn_image_params_t x_image;

  int32_t bias;
  uint16_t shift;
  int8_t scale;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

using AvgPoolGlobalThreadData = ElementwiseThreadData<AvgPoolGlobalArguments>;

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void avgpool_global_thread_worker(void* context) {
  auto* td = static_cast<AvgPoolGlobalThreadData*>(context);
  auto* args = td->args;
  avgpool2d_global_ext(args->Y, args->X, args->bias, args->scale, args->shift,
                       &args->x_image, td->start, td->element_count,
                       AVGPOOL2D_GLOBAL_FLAG_NONE);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

using AvgPoolGlobalOpData =
    MultiThreadedOpData<AvgPoolGlobalArguments, AvgPoolGlobalThreadData>;

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<AvgPoolGlobalOpData*>(node->user_data);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op_data->stack_size,
                                avgpool_global_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->threads.size(),
      &op_data->stack_scratch_index));

  const auto& input_shape = GetTensorShape(GetInput(context, node, 0));
  op_data->args.x_image = {(uint32_t)input_shape.Dims(1),
                           (uint32_t)input_shape.Dims(2),
                           (uint32_t)input_shape.Dims(3)};

  const auto* bss = GetInput(context, node, 1);
  op_data->args.bias = unpack<int32_t>(&bss->data.uint8[0]);
  op_data->args.shift = unpack<uint16_t>(&bss->data.uint8[5]);
  op_data->args.scale = unpack<int8_t>(&bss->data.uint8[4]);

  TFLITE_DCHECK(op_data->args.scale > 0);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<AvgPoolGlobalOpData*>(node->user_data);
  op_data->args.Y = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalOutput(context, node, 0));
  op_data->args.X = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalInput(context, node, 0));

  // initialize the threads
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  dispatcher->Initialize(avgpool_global_thread_worker, op_data->threads.size(),
                         op_data->stack_size, stack);

  void* dispatcher_args[op_data->threads.size()];
  for (int i = 0; i < op_data->threads.size(); i++) {
    dispatcher_args[i] = &op_data->threads[i];
  }

  dispatcher->Invoke(dispatcher_args, op_data->threads.size());

  return kTfLiteOk;
}

}  // namespace avgpool_global

TfLiteRegistration* Register_AvgPool2D_Global() {
  static TfLiteRegistration r = {
      ElementwiseInit<avgpool_global::AvgPoolGlobalOpData>, nullptr,
      avgpool_global::Prepare, avgpool_global::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
