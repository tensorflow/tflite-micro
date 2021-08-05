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

struct Work {
  int8_t* X;
  int8_t* Y;
  int8_t* scratch;
  nn::Filter2D* f;
};

ATTRIBUTE_THREAD_FUNCTION void conv2d_v2_thread_worker(void* context) {
  auto* work = static_cast<Work*>(context);
  work->f->execute(work->Y, work->X, work->scratch);
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
struct ThreadInfo {
  size_t stack_size;        // Each thread needs a stack
  size_t scratch_size;      // Each thread needs a scratch
  int stack_scratch_index;  // All threads stack and scratch consolidated into a
                            // single scratch buffer
  nn::Filter2D job;         // The job to be done by this thread
};

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct Conv2DOpData {
  size_t thread_count;
  ThreadInfo* threads;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);

  auto* op_data = construct_persistent_object<Conv2DOpData>(context);
  auto parser = CustomOptionParser(buffer, length);

  // op_data->thread_count = parser.parseNamedCustomOption("thread_count");

  // op_data->threads =
  // static_cast<ThreadInfo*>(context->AllocatePersistentBuffer(
  //     context, op_data->thread_count * sizeof(ThreadInfo)));

  for (int t = 0; t < op_data->thread_count; ++t) {
    // op_data->threads[t].scratch_size =
    //     parser.parseNamedCustomOption("scratch_size");

    // read the kernel type
    KernelType kt;  // = parser.parseNamedCustomOption("kernel_type");

    // std::vector<int> params_vector =
    //     parser.parseNamedCustomOption("params").asVector();
    switch (kt) {
      case Conv2dValidDirect_t:

        // nn::AbstractKernel::Params* akp =
        //     (nn::AbstractKernel::Params*)(buffer +
        //                                   abstract_kernel_params_buffer_idx);
        // nn::DerefInputFn::Params* dip =
        //     (nn::DerefInputFn::Params*)(buffer + memcpy_fn_buffer_idx);
        // nn::MatMulDirectFn::Params* mdp =
        //     (nn::MatMulDirectFn::Params*)(buffer + aggregate_fn_buffer_idx);
        // nn::OT_int8::Params* otp =
        //     (nn::OT_int8::Params*)(buffer + output_transform_fn_buffer_idx);

        // op_data->threads[t].job = new nn::Filter2D(akp, dip, );
        break;
      case Conv2dValidIndirect_t:
        break;
      case Conv2dPaddedInDirect_t:
        break;
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

    //#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME)
    op_data->threads[t].stack_size = require_stack;

    size_t request =
        op_data->threads[t].scratch_size + op_data->threads[t].stack_size;

    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, request, &op_data->threads[t].stack_scratch_index));
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);

  int n_threads = op_data->thread_count;
  /*
    auto* dispatcher = tflite::micro::xcore::GetDispatcher();

    dispatcher->Initialize(conv2d_v2_thread_worker, n_threads,
                           op_data->stack_size, stack);
    for (int t = 0; t < n_threads; ++t) {
      auto* stack = static_cast<char*>(context->GetScratchBuffer(
          context, op_data->threads[t].stack_scratch_index));

      TF_LITE_ENSURE(context, stack);
    }
  */
  // // initialize the threads

  // // TODO: move this to init
  // // create thread data
  // int n_th = op_data->execution_plan.GetNumThreads();
  // Conv2DThreadData thread_data[n_th];

  // for (int j{0}; j < n_th; j++) {
  //   thread_data[j].args = &op_data->args;
  // }

  // auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  // dispatcher->Initialize(Conv2DKernel<kernel_type>::get_worker(), n_th,
  //                        op_data->stack_size, stack);

  // const auto* weights = tflite::micro::GetEvalInput(context, node, 1);
  // const auto channel_size =
  //     Conv2DKernel<kernel_type>::calculate_output_channel_size(
  //         tflite::micro::GetTensorShape(weights));

  // const auto* weights_src_array =
  // tflite::micro::GetTensorData<int8_t>(weights); const auto* bso_src_array =
  // tflite::micro::GetTensorData<int8_t>(
  //     tflite::micro::GetEvalInput(context, node, 2));

  // if (kernel_type == Conv2DKernelType::kDepthwise) {
  //   if (op_data->weights_scratch_index >= 0) {
  //     op_data->args.depthwise_flags = CONV2D_DEPTHWISE_FLAG_SLICED_K;
  //   } else {
  //     op_data->args.depthwise_flags = (nn_conv2d_depthwise_flags_e)0;
  //     op_data->args.K = weights_src_array;
  //     op_data->args.BSO = (const nn_bso_block_t*)bso_src_array;
  //   }
  // }

  // size_t weights_src_offset = 0;
  // size_t biases_src_offset = 0;
  // for (int i_cg = 0; i_cg < op_data->execution_plan.changrps.size(); i_cg++)
  // {
  //   const auto& changrp = op_data->execution_plan.changrps[i_cg];

  //   // fetch weights and biases
  //   if (kernel_type == Conv2DKernelType::kDepthwise) {
  //     if (op_data->weights_scratch_index >= 0) {
  //       fetch_depthwise_subtensor((int8_t*)op_data->args.K,
  //       weights_src_array,
  //                                 op_data->args.window.shape.height,
  //                                 op_data->args.window.shape.width,
  //                                 channel_size, changrp.start, changrp.size);
  //     }
  //     if (op_data->bias_scratch_index >= 0) {
  //       FetchBuffer((int8_t**)&op_data->args.BSO,
  //                   &bso_src_array[biases_src_offset],
  //                   kBSOChannelGroupBytes);
  //       biases_src_offset += kBSOChannelGroupBytes;
  //     }
  //   } else {
  //     size_t weights_fetch_size = channel_size * changrp.size;
  //     FetchBuffer((int8_t**)&op_data->args.K,
  //                 &weights_src_array[weights_src_offset],
  //                 weights_fetch_size);
  //     weights_src_offset += weights_fetch_size;
  //     FetchBuffer((int8_t**)&op_data->args.BSO,
  //                 &bso_src_array[biases_src_offset], kBSOChannelGroupBytes);
  //     biases_src_offset += kBSOChannelGroupBytes;
  //   }

  //   // create tasks
  //   size_t n_rg = op_data->execution_plan.regions.size();
  //   void* dispatcher_args[n_rg];
  //   for (int i_rg = 0; i_rg < n_rg; i_rg++) {
  //     const auto& region = op_data->execution_plan.regions[i_rg];

  //     thread_data[i_rg].job = {{region.top, region.left, changrp.start},
  //                              {region.rows, region.cols, changrp.size}};
  //     dispatcher_args[i_rg] = &thread_data[i_rg];
  //   }
  //   dispatcher->Invoke(dispatcher_args, n_rg);
  // }

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
