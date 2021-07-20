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

#include "flatbuffers/flexbuffers.h"
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
namespace bconv {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct BConv2DArguments {
  const bnn_b32_t* X;
  const bnn_b32_t* K;

  nn_image_params_t x;
  nn_image_params_t y;
  nn_window_params_t k;

  union {
    bnn_b32_t* Y_bitpacked;
    int8_t* Y_int8;
  };

  union {
    const int32_t* thresholds;     // used in bitpacked only
    const int16_t* accu_modifier;  // used in generic int8 only
  };

  // for int8 only
  const int16_t* post_act_mult;
  const int16_t* post_act_bias;
  const output_transform_values_t* output_trf_parameters;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct BConv2DThreadData {
  // TODO: change this when new dispatcher is rolled out

  // This describes the region that that thread will process
  const RowColRegion* job;
  int thread_scratch_idx = -1;
  bnn_b32_t* thread_scratch;  // size should be K_h * K_w * C_in / 32 + 8
  const BConv2DArguments* args;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void bconv2d_bitpacked_deepin_thread_worker(
    void* context) {
  auto* td = static_cast<BConv2DThreadData*>(context);
  auto* args = td->args;
  auto* job = td->job;
  bconv2d_bin_DI_valid(args->Y_bitpacked, (const bnn_b256_t*)args->X,
                       (const bnn_b256_t*)args->K, args->thresholds, &args->x,
                       &args->y, &args->k, job->left, job->top, job->cols,
                       job->rows, 0, args->y.channels);
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_bitpacked_thread_worker(void* context) {
  auto* td = static_cast<BConv2DThreadData*>(context);
  auto* args = td->args;
  auto* job = td->job;
  bconv2d_bin_valid(args->Y_bitpacked, args->X, args->K, args->thresholds,
                    td->thread_scratch, &args->x, &args->y, &args->k, job->left,
                    job->top, job->cols, job->rows, 0, args->y.channels);
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_int8_deepin_deepout_thread_worker(
    void* context) {
  auto* td = static_cast<BConv2DThreadData*>(context);
  auto* args = td->args;
  auto* job = td->job;
  bconv2d_int8_DIDO_valid(args->Y_int8, (const bnn_b256_t*)args->X,
                          (const bnn_b256_t*)args->K, args->post_act_mult,
                          args->post_act_bias, args->output_trf_parameters,
                          &args->x, &args->y, &args->k, job->left, job->top,
                          job->cols, job->rows, 0, args->y.channels);
}

ATTRIBUTE_THREAD_FUNCTION void bconv2d_int8_thread_worker(void* context) {
  auto* td = static_cast<BConv2DThreadData*>(context);
  auto* args = td->args;
  auto* job = td->job;
  bconv2d_int8_valid(args->Y_int8, args->X, args->K, args->post_act_mult,
                     args->post_act_bias, args->accu_modifier,
                     args->output_trf_parameters, td->thread_scratch, &args->x,
                     &args->y, &td->args->k, job->left, job->top, job->cols,
                     job->rows, 0, args->y.channels);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct BConv2DOpData
    : MultiThreadedOpData<BConv2DArguments, BConv2DThreadData> {
  // The jobs (regions) threads will have to process.
  PersistentArray<RowColRegion> jobs;

  // TODO: remove this when better external memory handling is implemented
  // for loading from external mem
  int weights_scratch_idx = -1;
  int threshold_scratch_idx = -1;
  int bias_scratch_idx = -1;
  int multiplier_scratch_idx = -1;
  int accu_modifier_scratch_idx = -1;
  int output_trf_scratch_idx = -1;
};

// -------------------------------------------------------------------- //
// kernel types
// -------------------------------------------------------------------- //

enum class BConv2DKernelType {
  kBitpacked,
  kBitpackedDeepIn,
  kInt8,
  kInt8DeepInDeepOut,
};

template <BConv2DKernelType kernel_type>
struct BConv2DKernel {
  static inline const tflite::micro::xcore::ThreadFunction get_worker() {
    if (kernel_type == BConv2DKernelType::kBitpacked) {
      return bconv2d_bitpacked_thread_worker;
    } else if (kernel_type == BConv2DKernelType::kBitpackedDeepIn) {
      return bconv2d_bitpacked_deepin_thread_worker;
    } else if (kernel_type == BConv2DKernelType::kInt8) {
      return bconv2d_int8_thread_worker;
    } else if (kernel_type == BConv2DKernelType::kInt8DeepInDeepOut) {
      return bconv2d_int8_deepin_deepout_thread_worker;
    } else {
      UNSUPPORTED_KERNEL_TYPE(BConv2DKernelType);
    }
  };
  static inline void calculate_worker_stack_size(size_t& stack_size) {
    if (kernel_type == BConv2DKernelType::kBitpacked) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_bitpacked_thread_worker);
    } else if (kernel_type == BConv2DKernelType::kBitpackedDeepIn) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_bitpacked_deepin_thread_worker);
    } else if (kernel_type == BConv2DKernelType::kInt8) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size, bconv2d_int8_thread_worker);
    } else if (kernel_type == BConv2DKernelType::kInt8DeepInDeepOut) {
      GET_THREAD_FUNCTION_STACKSIZE(stack_size,
                                    bconv2d_int8_deepin_deepout_thread_worker);
    } else {
      UNSUPPORTED_KERNEL_TYPE(BConv2DKernelType);
    }
  };
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = construct_persistent_object<BConv2DOpData>(context);

  auto parser = CustomOptionParser(buffer, length);

  auto& k_shape = op_data->args.k.shape;
  parser.parseNamedTuple("K", op_data->args.y.channels, k_shape.height,
                         k_shape.width, op_data->args.x.channels);

  auto& k_stride = op_data->args.k.stride;
  parser.parseNamedTuple("stride", k_stride.vertical, k_stride.horizontal);

  // parse parallelization plan
  auto par_parser =
      CustomOptionParser(parser.parseNamedCustomOption("par").AsMap());

  auto regions = par_parser.parseNamedCustomOption("rc").AsVector();
  auto n_jobs = regions.size();
  op_data->jobs.allocate(context, n_jobs);
  for (int j{0}; j < n_jobs; j++) {
    auto region = regions[j].AsVector();
    op_data->jobs.append({region[0].AsInt32(), region[1].AsInt32(),
                          region[2].AsInt32(), region[3].AsInt32()});
  }

  auto n_threads = par_parser.parseNamedCustomOption("th").AsInt32();
  // TODO: remove this check when new dispatcher is rolled out
  TFLITE_CHECK_EQ(n_jobs, n_threads);
  op_data->threads.allocate(context, n_threads);
  BConv2DThreadData td;
  td.args = &op_data->args;
  for (int j{0}; j < n_threads; j++) {
    // TODO: remove this when new dispatcher is rolled out
    td.job = &op_data->jobs[j];
    op_data->threads.append(td);
  }

  op_data->args.k.dilation.horizontal = 1;
  op_data->args.k.dilation.vertical = 1;

  return op_data;
}

TfLiteStatus PrepareCommon(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto* op_data = reinterpret_cast<BConv2DOpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  op_data->args.x.height = (uint32_t)input->dims->data[1];
  op_data->args.x.width = (uint32_t)input->dims->data[2];

  const TfLiteTensor* output = GetOutput(context, node, 0);
  op_data->args.y.height = (uint32_t)output->dims->data[1];
  op_data->args.y.width = (uint32_t)output->dims->data[2];

  return kTfLiteOk;
}

template <BConv2DKernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PrepareCommon(context, node));

  auto* op_data = reinterpret_cast<BConv2DOpData*>(node->user_data);

  // TODO: fix this this when better weight fetching is implemented
  // allocate scratch buffers for input parameter tensors (if necessary)
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->weights_scratch_idx));

  if (kernel_type == BConv2DKernelType::kBitpacked ||
      kernel_type ==
          BConv2DKernelType::kBitpackedDeepIn) {  // output is bitpacked
    TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 2), op_data->threshold_scratch_idx));
  } else if (kernel_type == BConv2DKernelType::kInt8 ||
             kernel_type == BConv2DKernelType::kInt8DeepInDeepOut) {
    TF_LITE_ENSURE_EQ(context, NumInputs(node),
                      (kernel_type == BConv2DKernelType::kInt8) ? 6 : 5);
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 2), op_data->multiplier_scratch_idx));
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 3), op_data->bias_scratch_idx));
    TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
        context, GetInput(context, node, 4), op_data->output_trf_scratch_idx));
    if (kernel_type == BConv2DKernelType::kInt8) {
      TF_LITE_ENSURE_STATUS(
          request_scratch_if_needed(context, GetInput(context, node, 5),
                                    op_data->accu_modifier_scratch_idx));
    }
  } else {
    UNSUPPORTED_KERNEL_TYPE(BConv2DKernelType);
  }

  BConv2DKernel<kernel_type>::calculate_worker_stack_size(op_data->stack_size);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->threads.size(),
      &op_data->stack_scratch_index));

  if (kernel_type == BConv2DKernelType::kBitpacked ||
      kernel_type == BConv2DKernelType::kInt8) {
    int thread_scratch_size =
        4 * (op_data->args.k.shape.height * op_data->args.k.shape.width *
                 op_data->args.x.channels / XS1_ALL_BITS_SIZE +
             XS3_VPU_VREG_WIDTH_WORDS);

    for (auto& thread : op_data->threads) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, thread_scratch_size, &thread.thread_scratch_idx));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalCommon(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = reinterpret_cast<BConv2DOpData*>(node->user_data);
  op_data->args.X = tflite::micro::GetTensorData<bnn_b32_t>(
      tflite::micro::GetEvalInput(context, node, 0));

  TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
      context, op_data->args.K, tflite::micro::GetEvalInput(context, node, 1),
      op_data->weights_scratch_idx));

  return kTfLiteOk;
}

template <BConv2DKernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(EvalCommon(context, node));

  auto* op_data = reinterpret_cast<BConv2DOpData*>(node->user_data);

  if (kernel_type == BConv2DKernelType::kBitpacked ||
      kernel_type == BConv2DKernelType::kBitpackedDeepIn) {
    op_data->args.Y_bitpacked = tflite::micro::GetTensorData<bnn_b32_t>(
        tflite::micro::GetEvalOutput(context, node, 0));
    TF_LITE_ENSURE_STATUS(
        fetch_scratch_if_needed(context, op_data->args.thresholds,
                                tflite::micro::GetEvalInput(context, node, 2),
                                op_data->threshold_scratch_idx));
  } else if (kernel_type == BConv2DKernelType::kInt8 ||
             kernel_type == BConv2DKernelType::kInt8DeepInDeepOut) {
    op_data->args.Y_int8 = tflite::micro::GetTensorData<int8_t>(
        tflite::micro::GetEvalOutput(context, node, 0));

    TF_LITE_ENSURE_STATUS(
        fetch_scratch_if_needed(context, op_data->args.post_act_mult,
                                tflite::micro::GetEvalInput(context, node, 2),
                                op_data->multiplier_scratch_idx));
    TF_LITE_ENSURE_STATUS(
        fetch_scratch_if_needed(context, op_data->args.post_act_bias,
                                tflite::micro::GetEvalInput(context, node, 3),
                                op_data->bias_scratch_idx));
    TF_LITE_ENSURE_STATUS(
        fetch_scratch_if_needed(context, op_data->args.output_trf_parameters,
                                tflite::micro::GetEvalInput(context, node, 4),
                                op_data->output_trf_scratch_idx));
    if (kernel_type == BConv2DKernelType::kInt8) {
      TF_LITE_ENSURE_STATUS(
          fetch_scratch_if_needed(context, op_data->args.accu_modifier,
                                  tflite::micro::GetEvalInput(context, node, 5),
                                  op_data->accu_modifier_scratch_idx));
    }
  } else {
    UNSUPPORTED_KERNEL_TYPE(BConv2DKernelType);
  }

  // initialize the threads
  auto* stack = static_cast<char*>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);

  auto* dispatcher = tflite::micro::xcore::GetDispatcher();
  dispatcher->Initialize(BConv2DKernel<kernel_type>::get_worker(),
                         op_data->threads.size(), op_data->stack_size, stack);

  // start threads
  int i_arg = 0;
  int n_arg = op_data->threads.size();
  void* dispatcher_args[n_arg];
  for (auto& thread : op_data->threads) {
    if (kernel_type == BConv2DKernelType::kBitpacked ||
        kernel_type == BConv2DKernelType::kInt8) {
      thread.thread_scratch = static_cast<bnn_b32_t*>(
          context->GetScratchBuffer(context, thread.thread_scratch_idx));
    }
    dispatcher_args[i_arg++] = reinterpret_cast<void*>(&thread);
  }
  dispatcher->Invoke(dispatcher_args, n_arg);

  return kTfLiteOk;
}

}  // namespace bconv

TfLiteRegistration* Register_BConv2D_Bitpacked_Deepin() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr,
      bconv::Prepare<bconv::BConv2DKernelType::kBitpackedDeepIn>,
      bconv::Eval<bconv::BConv2DKernelType::kBitpackedDeepIn>};
  return &r;
}

TfLiteRegistration* Register_BConv2D_Bitpacked() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr,
      bconv::Prepare<bconv::BConv2DKernelType::kBitpacked>,
      bconv::Eval<bconv::BConv2DKernelType::kBitpacked>};
  return &r;
}

TfLiteRegistration* Register_BConv2D_Int8_Deepin_Deepout() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr,
      bconv::Prepare<bconv::BConv2DKernelType::kInt8DeepInDeepOut>,
      bconv::Eval<bconv::BConv2DKernelType::kInt8DeepInDeepOut>};
  return &r;
}

TfLiteRegistration* Register_BConv2D_Int8() {
  static TfLiteRegistration r = {
      bconv::Init, nullptr, bconv::Prepare<bconv::BConv2DKernelType::kInt8>,
      bconv::Eval<bconv::BConv2DKernelType::kInt8>};
  return &r;
}

}  // namespace xcore

}  // namespace micro
}  // namespace ops
}  // namespace tflite
