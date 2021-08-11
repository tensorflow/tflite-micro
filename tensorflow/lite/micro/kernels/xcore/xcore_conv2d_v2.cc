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

#include "Conv2d.hpp"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_custom_options.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_dispatcher.h"
#include "tensorflow/lite/micro/kernels/xcore/xcore_op_utils.h"
extern "C" {
#include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv_v2 {

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DThread {
  int8_t* X;
  int8_t* Y;
  int8_t* scratch;
  nn::Filter2D* f;
};

extern "C" {
#pragma stackfunction 1000
ATTRIBUTE_THREAD_FUNCTION
void conv2d_v2_thread_worker(void* context) {
  auto work = static_cast<Conv2DThread*>(context);
  work->f->execute(work->Y, work->X, work->scratch);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

enum KernelType {
  Conv2dValidDirect_t,
  Conv2dValidIndirect_t,
  Conv2dPaddedInDirect_t,
};

/**
 * @brief This describes the memory requirements of a worker thread. It also
 * includes an array of the work to be done by said worker.
 *
 */
struct Conv2DThreadInfo {
  size_t stack_size;        // Each thread needs a stack
  size_t scratch_size;      // Each thread needs a scratch
  int stack_scratch_index;  // All threads stack and scratch consolidated into a
                            // single scratch buffer
  nn::Filter2D* filter2D;   // The job to be done by this thread
};

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct Conv2DOpData {
  size_t thread_count;
  Conv2DThreadInfo* threads;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

template <typename T, bool pointer_serialization>
T* getDeserializedParams(TfLiteContext* context, std::string str) {
  char* allocated_memory;
  int allocationByteCount =
      T::get_allocation_byte_count(str.c_str()) + sizeof(T);
  allocated_memory =
      (char*)context->AllocatePersistentBuffer(context, allocationByteCount);
  T* param = T::template deserialise<T>(allocated_memory, str.c_str());
  return param;
}

template <typename T>
T* getDeserializedParams(TfLiteContext* context, std::string str) {
  char* allocated_memory;
  int allocationByteCount = sizeof(T);
  allocated_memory =
      (char*)context->AllocatePersistentBuffer(context, allocationByteCount);
  T* param = T::template deserialise<T>(allocated_memory, str.c_str());
  return param;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);

  auto op_data = construct_persistent_object<Conv2DOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  auto threads = parser.parseNamedCustomOption("threads").AsVector();
  auto thread_count = threads.size();
  op_data->thread_count = thread_count;
  op_data->threads =
      static_cast<Conv2DThreadInfo*>(context->AllocatePersistentBuffer(
          context, op_data->thread_count * sizeof(Conv2DThreadInfo)));

  for (int t = 0; t < op_data->thread_count; ++t) {
    flexbuffers::Vector params = threads[t].AsVector();
    op_data->threads[t].scratch_size = params[0].AsInt32();
    // read the kernel type
    KernelType kt = (KernelType)params[1].AsInt32();

    switch (kt) {
      case Conv2dValidDirect_t: {
        nn::Filter2D::Params* ak_params =
            getDeserializedParams<nn::Filter2D::Params>(
                context, params[2].As<std::string>());
        nn::DerefInputFn::Params* mf_params =
            getDeserializedParams<nn::DerefInputFn::Params>(
                context, params[3].As<std::string>());
        nn::MatMulDirectFn::Params* af_params =
            getDeserializedParams<nn::MatMulDirectFn::Params, true>(
                context, params[4].As<std::string>());
        nn::OT_int8::Params* ot_params =
            getDeserializedParams<nn::OT_int8::Params, true>(
                context, params[5].As<std::string>());

        auto memcpy = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::DerefInputFn))) nn::DerefInputFn(mf_params);

        auto aggregator = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::MatMulDirectFn))) nn::MatMulDirectFn(af_params);

        auto ot = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::OT_int8))) nn::OT_int8(ot_params);

        auto conv2d = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::Conv2dValidDirect)))
            nn::Conv2dValidDirect(ak_params, memcpy, aggregator, ot);

        op_data->threads[t].filter2D = conv2d;
      } break;
      case Conv2dValidIndirect_t: {
        nn::Filter2D::Params* ak_params =
            getDeserializedParams<nn::Filter2D::Params>(
                context, params[2].As<std::string>());
        nn::ImToColValid::Params* mf_params =
            getDeserializedParams<nn::ImToColValid::Params>(
                context, params[3].As<std::string>());
        nn::MatMulInt8::Params* af_params =
            getDeserializedParams<nn::MatMulInt8::Params, true>(
                context, params[4].As<std::string>());
        nn::OT_int8::Params* ot_params =
            getDeserializedParams<nn::OT_int8::Params, true>(
                context, params[5].As<std::string>());

        auto memcpy = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::ImToColValid))) nn::ImToColValid(mf_params);

        auto aggregator = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::MatMulInt8))) nn::MatMulInt8(af_params);

        auto ot = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::OT_int8))) nn::OT_int8(ot_params);

        auto conv2d = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::Conv2dValidIndirect)))
            nn::Conv2dValidIndirect(ak_params, memcpy, aggregator, ot);

        op_data->threads[t].filter2D = conv2d;
      } break;
      case Conv2dPaddedInDirect_t: {
        nn::Filter2D::Params* ak_params =
            getDeserializedParams<nn::Filter2D::Params>(
                context, params[2].As<std::string>());
        nn::ImToColPadded::Params* mf_params =
            getDeserializedParams<nn::ImToColPadded::Params>(
                context, params[3].As<std::string>());
        nn::MatMulInt8::Params* af_params =
            getDeserializedParams<nn::MatMulInt8::Params, true>(
                context, params[4].As<std::string>());
        nn::OT_int8::Params* ot_params =
            getDeserializedParams<nn::OT_int8::Params, true>(
                context, params[5].As<std::string>());

        auto memcpy = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::ImToColPadded))) nn::ImToColPadded(mf_params);

        auto aggregator = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::MatMulInt8))) nn::MatMulInt8(af_params);

        auto ot = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::OT_int8))) nn::OT_int8(ot_params);

        auto conv2d = new (context->AllocatePersistentBuffer(
            context, sizeof(nn::Conv2dPaddedInDirect)))
            nn::Conv2dPaddedInDirect(ak_params, memcpy, aggregator, ot);

        op_data->threads[t].filter2D = conv2d;
      } break;
    }
  }
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  // TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);
  for (int t = 0; t < op_data->thread_count; ++t) {
    // allocate the stack for thread workers
    size_t require_stack;
    // get stack size
    GET_THREAD_FUNCTION_STACKSIZE(require_stack, conv2d_v2_thread_worker);
    op_data->threads[t].stack_size = require_stack;
    size_t request =
        op_data->threads[t].scratch_size + op_data->threads[t].stack_size;
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, request, &op_data->threads[t].stack_scratch_index));
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);
  int n_threads = op_data->thread_count;
  auto* dispatcher = tflite::micro::xcore::GetDispatcher();

  for (int t = 0; t < n_threads; ++t) {
    auto* stack = static_cast<char*>(context->GetScratchBuffer(
        context, op_data->threads[t].stack_scratch_index));
    TF_LITE_ENSURE(context, stack);

    dispatcher->Initialize(conv2d_v2_thread_worker, n_threads,
                           op_data->threads[t].stack_size, stack);
  }

  Conv2DThread thread_data[n_threads];
  void* dispatcher_args[n_threads];
  for (int t = 0; t < n_threads; ++t) {
    thread_data[t].X = (int8_t*)tflite::micro::GetTensorData<int8_t>(input);
    thread_data[t].Y = (int8_t*)tflite::micro::GetTensorData<int8_t>(output);
    thread_data[t].scratch = (int8_t*)context->GetScratchBuffer(
        context, op_data->threads[t].stack_scratch_index);
    thread_data[t].f = op_data->threads[t].filter2D;
    dispatcher_args[t] = reinterpret_cast<void*>(&thread_data[t]);
  }

  dispatcher->Invoke(dispatcher_args, n_threads);

  return kTfLiteOk;
}

}  // namespace conv_v2

TfLiteRegistration* Register_Conv2D_V2() {
  static TfLiteRegistration r = {conv_v2::Init, nullptr, conv_v2::Prepare,
                                 conv_v2::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
