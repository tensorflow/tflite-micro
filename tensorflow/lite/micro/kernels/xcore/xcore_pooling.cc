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
namespace pooling {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct Pool2DArguments {
  nn_image_t* Y;
  const nn_image_t* X;

  nn_image_params_t x_image;
  nn_image_params_t y_image;
  nn_window_params_t window;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //
struct Pool2DThreadData {
  Pool2DArguments* args;
  nn_window_op_job_params_t job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void maxpool_thread_worker(void* context) {
  auto* td = static_cast<Pool2DThreadData*>(context);
  auto* args = td->args;
  maxpool2d_ext(args->Y, args->X, &args->x_image, &args->y_image, &args->window,
                &td->job, MAXPOOL2D_FLAG_NONE);
}

ATTRIBUTE_THREAD_FUNCTION void avgpool_thread_worker(void* context) {
  auto* td = static_cast<Pool2DThreadData*>(context);
  auto* args = td->args;
  avgpool2d_ext(args->Y, args->X, &args->x_image, &args->y_image, &args->window,
                &td->job, AVGPOOL2D_FLAG_NONE);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct PoolOpData {
  Pool2DArguments args;
  ExecutionPlan execution_plan;
  int stack_scratch_index = -1;
  size_t stack_size;
};

// -------------------------------------------------------------------- //
// kernel types
// -------------------------------------------------------------------- //

enum class Pool2DKernelType {
  kMaxPool2D,
  kAvgPool2D,
};

template <Pool2DKernelType kernel_type>
struct Pool2DKernel {
  static inline const tflite::micro::xcore::ThreadFunction get_worker() {
    if (kernel_type == Pool2DKernelType::kMaxPool2D) {
      return maxpool_thread_worker;
    } else if (kernel_type == Pool2DKernelType::kAvgPool2D) {
      return avgpool_thread_worker;
    } else {
      UNSUPPORTED_KERNEL_TYPE(Pool2DKernelType);
    }
  };
  static inline void calculate_worker_stack_size(size_t& stack_size) {
    if (kernel_type == Pool2DKernelType::kMaxPool2D) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, maxpool_thread_worker);
    } else if (kernel_type == Pool2DKernelType::kAvgPool2D) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, avgpool_thread_worker);
    } else {
      UNSUPPORTED_KERNEL_TYPE(Pool2DKernelType);
    }
  };
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = construct_persistent_object<PoolOpData>(context);

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op_data->execution_plan);

  auto parser = CustomOptionParser(buffer, length);

  auto& k_stride = op_data->args.window.stride;
  parser.parseNamedTuple("stride", k_stride.vertical, k_stride.horizontal);
  auto& k_shape = op_data->args.window.shape;
  parser.parseNamedTuple("pool", k_shape.height, k_shape.width);
  op_data->args.window.start = {0, 0};

  return op_data;
}

TfLiteStatus PrepareCommon(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<PoolOpData*>(node->user_data);

  const auto& input_shape = GetTensorShape(GetInput(context, node, 0));
  op_data->args.x_image = {(uint32_t)input_shape.Dims(1),
                           (uint32_t)input_shape.Dims(2),
                           (uint32_t)input_shape.Dims(3)};

  const auto& output_shape = GetTensorShape(GetOutput(context, node, 0));
  op_data->args.y_image = {(uint32_t)output_shape.Dims(1),
                           (uint32_t)output_shape.Dims(2),
                           (uint32_t)output_shape.Dims(3)};

  return kTfLiteOk;
}

template <Pool2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto* op_data = reinterpret_cast<PoolOpData*>(node->user_data);

  // allocate the stack for thread workers
  Pool2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->execution_plan.GetNumThreads(),
      &op_data->stack_scratch_index));

  return kTfLiteOk;
}

TfLiteStatus EvalCommon(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<PoolOpData*>(node->user_data);
  op_data->args.Y = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalOutput(context, node, 0));
  op_data->args.X = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalInput(context, node, 0));
  return kTfLiteOk;
}

template <Pool2DKernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  auto* op_data = reinterpret_cast<PoolOpData*>(node->user_data);

  // initialize the threads
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  // TODO: move this to init
  // create thread data
  int n_th = op_data->execution_plan.GetNumThreads();
  Pool2DThreadData thread_data[n_th];
  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op_data->args;
  }

  auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  dispatcher->Initialize(Pool2DKernel<kernel_type>::get_worker(), n_th,
                         op_data->stack_size, stack);

  // create tasks
  size_t n_rg = op_data->execution_plan.regions.size();
  void* dispatcher_args[n_rg];
  for (int i_rg = 0; i_rg < n_rg; i_rg++) {
    const RowColRegion& region = op_data->execution_plan.regions[i_rg];
    thread_data[i_rg].job = {
        {region.top, region.left, 0},
        {region.rows, region.cols, (int32_t)op_data->args.y_image.channels}};

    dispatcher_args[i_rg] = &thread_data[i_rg];
  }
  dispatcher->Invoke(dispatcher_args, n_rg);

  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_MaxPool2D() {
  static TfLiteRegistration r = {
      pooling::Init, nullptr,
      pooling::Prepare<pooling::Pool2DKernelType::kMaxPool2D>,
      pooling::Eval<pooling::Pool2DKernelType::kMaxPool2D>};
  return &r;
}

TfLiteRegistration* Register_AvgPool2D() {
  static TfLiteRegistration r = {
      pooling::Init, nullptr,
      pooling::Prepare<pooling::Pool2DKernelType::kAvgPool2D>,
      pooling::Eval<pooling::Pool2DKernelType::kAvgPool2D>};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
