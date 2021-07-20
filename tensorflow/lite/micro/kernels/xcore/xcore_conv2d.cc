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
namespace conv {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct Conv2DArguments {
  nn_image_t* Y;
  const nn_image_t* X;
  const nn_tensor_t* K;
  const nn_bso_block_t* BSO;

  nn_image_params_t x_image;
  nn_image_params_t y_image;

  nn_window_params_t window;                    // not used by 1x1
  int8_t zero_point;                            // not used by 1x1
  nn_conv2d_depthwise_flags_e depthwise_flags;  // only for depthwise
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DThreadData {
  Conv2DArguments* args;
  nn_window_op_job_params_t job;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void conv2d_shallow_thread_worker(void* context) {
  auto* td = static_cast<Conv2DThreadData*>(context);
  auto* args = td->args;
  conv2d_shallowin_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                       &args->x_image, &args->y_image, &args->window, &td->job,
                       CONV2D_SHALLOWIN_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_deep_thread_worker(void* context) {
  auto* td = static_cast<Conv2DThreadData*>(context);
  auto* args = td->args;

  conv2d_deep_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                  &args->x_image, &args->y_image, &args->window, &td->job,
                  CONV2D_DEEP_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_1x1_thread_worker(void* context) {
  auto* td = static_cast<Conv2DThreadData*>(context);
  auto* args = td->args;

  // TODO: consider changing the kernel to unify this job struct
  nn_conv2d_1x1_job_params_t job;
  job.start = td->job.start;
  job.size.channels = td->job.size.channels;
  job.size.pixels = td->job.size.rows * td->job.size.cols;
  conv2d_1x1_ext(args->Y, args->X, args->K, args->BSO, &args->x_image,
                 &args->y_image, &job, CONV2D_1X1_FLAG_SLICED_K);
}

ATTRIBUTE_THREAD_FUNCTION void conv2d_depthwise_thread_worker(void* context) {
  auto* td = static_cast<Conv2DThreadData*>(context);
  auto* args = td->args;
  conv2d_depthwise_ext(args->Y, args->X, args->K, args->BSO, args->zero_point,
                       &args->x_image, &args->y_image, &args->window, &td->job,
                       args->depthwise_flags);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct Conv2DOpData {
  Conv2DArguments args;
  ExecutionPlan execution_plan;
  int stack_scratch_index = -1;
  size_t stack_size;
  int weights_scratch_index = -1;
  int bias_scratch_index = -1;
};

// -------------------------------------------------------------------- //
// kernel types
// -------------------------------------------------------------------- //

enum class Conv2DKernelType {
  kDeep,
  kShallow,
  kOneByOne,
  kDepthwise,
};

template <Conv2DKernelType kernel_type>
struct Conv2DKernel {
  static inline const tflite::micro::xcore::ThreadFunction get_worker() {
    if (kernel_type == Conv2DKernelType::kDeep) {
      return conv2d_deep_thread_worker;
    } else if (kernel_type == Conv2DKernelType::kShallow) {
      return conv2d_shallow_thread_worker;
    } else if (kernel_type == Conv2DKernelType::kOneByOne) {
      return conv2d_1x1_thread_worker;
    } else if (kernel_type == Conv2DKernelType::kDepthwise) {
      return conv2d_depthwise_thread_worker;
    } else {
      UNSUPPORTED_KERNEL_TYPE(Conv2DKernelType);
    }
  };
  static inline void calculate_worker_stack_size(size_t& stack_size) {
    if (kernel_type == Conv2DKernelType::kDeep) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_deep_thread_worker);
    } else if (kernel_type == Conv2DKernelType::kShallow) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_shallow_thread_worker);
    } else if (kernel_type == Conv2DKernelType::kOneByOne) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_1x1_thread_worker);
    } else if (kernel_type == Conv2DKernelType::kDepthwise) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, conv2d_depthwise_thread_worker);
    } else {
      UNSUPPORTED_KERNEL_TYPE(Conv2DKernelType);
    }
  };
  static inline size_t calculate_output_channel_size(
      const tflite::RuntimeShape& weights_shape) {
    if (kernel_type == Conv2DKernelType::kDeep ||
        kernel_type == Conv2DKernelType::kShallow) {
      return weights_shape.Dims(1) * weights_shape.Dims(2) *
             weights_shape.Dims(3);
    } else if (kernel_type == Conv2DKernelType::kOneByOne) {
      return weights_shape.Dims(1);
    } else if (kernel_type == Conv2DKernelType::kDepthwise) {
      return weights_shape.Dims(2);
    } else {
      UNSUPPORTED_KERNEL_TYPE(Conv2DKernelType);
    }
  };
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void* InitCommon(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = construct_persistent_object<Conv2DOpData>(context);

  // parse custom options
  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length,
                       &op_data->execution_plan);  // TODO deprecate this

  return op_data;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data =
      static_cast<Conv2DOpData*>(InitCommon(context, buffer, length));

  auto parser = CustomOptionParser(buffer, length);
  auto& k_stride = op_data->args.window.stride;
  parser.parseNamedTuple("stride", k_stride.vertical, k_stride.horizontal);

  auto& start_offsets = op_data->args.window.start;
  parser.parseNamedTuple("pad", start_offsets.row, start_offsets.column);

  // TODO: consider templating for kernel types and do this for shallow only
  auto k_w = parser.parseNamedCustomOption("Kw");
  if (!k_w.IsNull()) {
    op_data->args.window.shape.width = k_w.AsInt32();
  }

  return op_data;
}

TfLiteStatus PrepareCommon(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);

  TF_LITE_ENSURE_STATUS(
      request_scratch_if_needed(context, GetInput(context, node, 1)->data.data,
                                op_data->execution_plan.GetWeightsScratchSize(),
                                op_data->weights_scratch_index));
  TF_LITE_ENSURE_STATUS(
      request_scratch_if_needed(context, GetInput(context, node, 2)->data.data,
                                op_data->execution_plan.GetBiasScratchSize(),
                                op_data->bias_scratch_index));

  const auto* input = GetInput(context, node, 0);
  op_data->args.zero_point = input->params.zero_point;
  const auto& input_shape = GetTensorShape(input);
  op_data->args.x_image = {(uint32_t)input_shape.Dims(1),
                           (uint32_t)input_shape.Dims(2),
                           (uint32_t)input_shape.Dims(3)};

  const auto& output_shape = GetTensorShape(GetOutput(context, node, 0));
  op_data->args.y_image = {(uint32_t)output_shape.Dims(1),
                           (uint32_t)output_shape.Dims(2),
                           (uint32_t)output_shape.Dims(3)};

  return kTfLiteOk;
}

template <Conv2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);

  // allocate the stack for thread workers
  Conv2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->execution_plan.regions.size(),
      &op_data->stack_scratch_index));

  const auto& weight_shape = GetTensorShape(GetInput(context, node, 1));
  if (kernel_type == Conv2DKernelType::kOneByOne) {
    op_data->args.window.shape = {1, 1};
  } else if (kernel_type == Conv2DKernelType::kShallow) {
    // NOTE: width is already set in init
    op_data->args.window.shape.height = weight_shape.Dims(1);
  } else if (kernel_type == Conv2DKernelType::kDepthwise) {
    op_data->args.window.shape.height = weight_shape.Dims(0);
    op_data->args.window.shape.width = weight_shape.Dims(1);
  } else if (kernel_type == Conv2DKernelType::kDeep) {
    op_data->args.window.shape.height = weight_shape.Dims(1);
    op_data->args.window.shape.width = weight_shape.Dims(2);
  }

  return kTfLiteOk;
}

TfLiteStatus EvalCommon(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);
  op_data->args.Y = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalOutput(context, node, 0));
  op_data->args.X = tflite::micro::GetTensorData<nn_image_t>(
      tflite::micro::GetEvalInput(context, node, 0));

  if (op_data->weights_scratch_index >= 0) {
    op_data->args.K = static_cast<nn_tensor_t*>(
        context->GetScratchBuffer(context, op_data->weights_scratch_index));
    TFLITE_DCHECK(op_data->args.K != nullptr);
  }
  if (op_data->bias_scratch_index >= 0) {
    op_data->args.BSO = static_cast<nn_bso_block_t*>(
        context->GetScratchBuffer(context, op_data->bias_scratch_index));
    TFLITE_DCHECK(op_data->args.BSO != nullptr);
  }
  return kTfLiteOk;
}

static void fetch_depthwise_subtensor(int8_t* dest, const nn_tensor_t* weights,
                                      const unsigned K_h, const unsigned K_w,
                                      const unsigned X_c,
                                      const unsigned start_channel,
                                      const unsigned channel_count) {
  assert(start_channel % 16 == 0);
  assert(channel_count % 4 == 0);

  weights =
      &(weights[start_channel]);  // Address of weights[0][0][start_channel]

  // Total of K_h * K_w blocks, for a total of K_h*K_w*channel_count bytes
  for (int k = 0; k < K_h * K_w; k++) {
    FetchBuffer(&dest, weights, channel_count);
    // memcpy(dest, weights, channel_count);
    dest = &(dest[channel_count]);
    weights = &(weights[X_c]);
  }
}

template <Conv2DKernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  auto* op_data = reinterpret_cast<Conv2DOpData*>(node->user_data);

  // initialize the threads
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  // TODO: move this to init
  // create thread data
  int n_th = op_data->execution_plan.GetNumThreads();
  Conv2DThreadData thread_data[n_th];

  for (int j{0}; j < n_th; j++) {
    thread_data[j].args = &op_data->args;
  }

  auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  dispatcher->Initialize(Conv2DKernel<kernel_type>::get_worker(), n_th,
                         op_data->stack_size, stack);

  const auto* weights = tflite::micro::GetEvalInput(context, node, 1);
  const auto channel_size =
      Conv2DKernel<kernel_type>::calculate_output_channel_size(
          tflite::micro::GetTensorShape(weights));

  const auto* weights_src_array = tflite::micro::GetTensorData<int8_t>(weights);
  const auto* bso_src_array = tflite::micro::GetTensorData<int8_t>(
      tflite::micro::GetEvalInput(context, node, 2));

  if (kernel_type == Conv2DKernelType::kDepthwise) {
    if (op_data->weights_scratch_index >= 0) {
      op_data->args.depthwise_flags = CONV2D_DEPTHWISE_FLAG_SLICED_K;
    } else {
      op_data->args.depthwise_flags = (nn_conv2d_depthwise_flags_e)0;
      op_data->args.K = weights_src_array;
      op_data->args.BSO = (const nn_bso_block_t*)bso_src_array;
    }
  }

  size_t weights_src_offset = 0;
  size_t biases_src_offset = 0;
  for (int i_cg = 0; i_cg < op_data->execution_plan.changrps.size(); i_cg++) {
    const auto& changrp = op_data->execution_plan.changrps[i_cg];

    // fetch weights and biases
    if (kernel_type == Conv2DKernelType::kDepthwise) {
      if (op_data->weights_scratch_index >= 0) {
        fetch_depthwise_subtensor((int8_t*)op_data->args.K, weights_src_array,
                                  op_data->args.window.shape.height,
                                  op_data->args.window.shape.width,
                                  channel_size, changrp.start, changrp.size);
      }
      if (op_data->bias_scratch_index >= 0) {
        FetchBuffer((int8_t**)&op_data->args.BSO,
                    &bso_src_array[biases_src_offset], kBSOChannelGroupBytes);
        biases_src_offset += kBSOChannelGroupBytes;
      }
    } else {
      size_t weights_fetch_size = channel_size * changrp.size;
      FetchBuffer((int8_t**)&op_data->args.K,
                  &weights_src_array[weights_src_offset], weights_fetch_size);
      weights_src_offset += weights_fetch_size;
      FetchBuffer((int8_t**)&op_data->args.BSO,
                  &bso_src_array[biases_src_offset], kBSOChannelGroupBytes);
      biases_src_offset += kBSOChannelGroupBytes;
    }

    // create tasks
    size_t n_rg = op_data->execution_plan.regions.size();
    void* dispatcher_args[n_rg];
    for (int i_rg = 0; i_rg < n_rg; i_rg++) {
      const auto& region = op_data->execution_plan.regions[i_rg];

      thread_data[i_rg].job = {{region.top, region.left, changrp.start},
                               {region.rows, region.cols, changrp.size}};
      dispatcher_args[i_rg] = &thread_data[i_rg];
    }
    dispatcher->Invoke(dispatcher_args, n_rg);
  }

  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_Conv2D_Deep() {
  static TfLiteRegistration r = {conv::Init, nullptr,
                                 conv::Prepare<conv::Conv2DKernelType::kDeep>,
                                 conv::Eval<conv::Conv2DKernelType::kDeep>};
  return &r;
}

TfLiteRegistration* Register_Conv2D_Shallow() {
  static TfLiteRegistration r = {
      conv::Init, nullptr, conv::Prepare<conv::Conv2DKernelType::kShallow>,
      conv::Eval<conv::Conv2DKernelType::kShallow>};
  return &r;
}

TfLiteRegistration* Register_Conv2D_1x1() {
  static TfLiteRegistration r = {
      conv::InitCommon, nullptr,
      conv::Prepare<conv::Conv2DKernelType::kOneByOne>,
      conv::Eval<conv::Conv2DKernelType::kOneByOne>};
  return &r;
}

TfLiteRegistration* Register_Conv2D_Depthwise() {
  static TfLiteRegistration r = {
      conv::Init, nullptr, conv::Prepare<conv::Conv2DKernelType::kDepthwise>,
      conv::Eval<conv::Conv2DKernelType::kDepthwise>};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
