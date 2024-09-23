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

#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/kernels/batch_matmul.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node)
      : params(static_cast<TfLiteBatchMatMulParams*>(node->builtin_data)),
        op_data(static_cast<OpDataBatchMatmul*>(node->user_data)) {}

  TfLiteBatchMatMulParams* params;
  OpDataBatchMatmul* op_data;
};

struct PrepareOpContext : OpContext {
  PrepareOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node),
        micro_context_(GetMicroContext(context)),
        lhs(micro_context_->AllocateTempInputTensor(
            node, kBatchMatmulInputLhsTensor)),
        rhs(micro_context_->AllocateTempInputTensor(
            node, kBatchMatmulInputRhsTensor)),
        output(micro_context_->AllocateTempOutputTensor(
            node, kBatchMatmulOutputTensor)) {}

  ~PrepareOpContext() {
    if (lhs != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(lhs);
    }
    if (rhs != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(rhs);
    }
    if (output != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(output);
    }
  }

 private:
  MicroContext* micro_context_;

 public:
  TfLiteTensor* lhs;
  TfLiteTensor* rhs;
  TfLiteTensor* output;
};

struct EvalOpContext : OpContext {
  EvalOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node),
        lhs(tflite::micro::GetEvalInput(context, node,
                                        kBatchMatmulInputLhsTensor)),
        rhs(tflite::micro::GetEvalInput(context, node,
                                        kBatchMatmulInputRhsTensor)),
        output(tflite::micro::GetEvalOutput(context, node,
                                            kBatchMatmulOutputTensor)) {}

  const TfLiteEvalTensor* lhs;
  const TfLiteEvalTensor* rhs;
  TfLiteEvalTensor* output;
};

TfLiteEvalTensor* AllocInitTransposeTensorFromTfLiteTensor(
    TfLiteContext* context, const TfLiteTensor& tensor) {
  MicroContext* micro_context = GetMicroContext(context);
  TfLiteEvalTensor* eval_tensor = static_cast<TfLiteEvalTensor*>(
      micro_context->AllocatePersistentBuffer(sizeof(TfLiteEvalTensor)));
  if (eval_tensor == nullptr) {
    return nullptr;
  }

  eval_tensor->type = tensor.type;

  const int tensor_rank = NumDimensions(&tensor);
  const size_t eval_dims_size = TfLiteIntArrayGetSizeInBytes(tensor_rank);
  eval_tensor->dims = static_cast<TfLiteIntArray*>(
      micro_context->AllocatePersistentBuffer(eval_dims_size));
  if (eval_tensor->dims == nullptr) {
    return nullptr;
  }
  eval_tensor->dims->size = tensor_rank;
  for (int i = 0; i < tensor_rank - 2; ++i) {
    eval_tensor->dims->data[i] = tensor.dims->data[i];
  }
  // Swap last two dimensions.
  eval_tensor->dims->data[tensor_rank - 2] = tensor.dims->data[tensor_rank - 1];
  eval_tensor->dims->data[tensor_rank - 1] = tensor.dims->data[tensor_rank - 2];

  const size_t eval_data_size = static_cast<size_t>(NumElements(&tensor)) *
                                TfLiteTypeGetSize(tensor.type);
  eval_tensor->data.data =
      micro_context->AllocatePersistentBuffer(eval_data_size);
  if (eval_tensor->data.data == nullptr) {
    return nullptr;
  }

  return eval_tensor;
}

// Initializes tensors to store transposed operands.
// Allocate storage for hybrid quantization if needed.
// Allocate normal quantization data if needed.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   const PrepareOpContext& op_context) {
  OpDataBatchMatmul* op_data = op_context.op_data;
  const TfLiteTensor* lhs = op_context.lhs;
  const TfLiteTensor* rhs = op_context.rhs;
  MicroContext* micro_context = GetMicroContext(context);

  op_data->quantization = nullptr;
  op_data->lhs_transposed_tensor = nullptr;
  op_data->rhs_transposed_tensor = nullptr;

  if (lhs->type == kTfLiteInt8 || lhs->type == kTfLiteInt16) {
    op_data->quantization = static_cast<decltype(op_data->quantization)>(
        micro_context->AllocatePersistentBuffer(
            sizeof(*op_data->quantization)));
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
  }

  // tensor for Transposed LHS;
  if (op_context.params->adj_x) {
    op_data->lhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *lhs);
    TF_LITE_ENSURE(context, op_data->lhs_transposed_tensor != nullptr);
  }

  // We need a buffer for the RHS if we need to transpose the RHS. We
  // transpose by default, so that the two inputs (LHS and RHS) are in a proper
  // layout for our fast matrix multiplication routines. If the transpose flag
  // is set by the caller, the data is already in the desired layout.
  if (!op_context.params->adj_y) {
    op_data->rhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *rhs);
    TF_LITE_ENSURE(context, op_data->rhs_transposed_tensor != nullptr);
  }

  return kTfLiteOk;
}

void* BatchMatMulInit(TfLiteContext* context, const char* buffer,
                      size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  MicroContext* micro_context = GetMicroContext(context);
  return micro_context->AllocatePersistentBuffer(sizeof(OpDataBatchMatmul));
}

TfLiteStatus BatchMatMulPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PrepareOpContext op_context(context, node);
  const TfLiteTensor* lhs_data = op_context.lhs;
  TF_LITE_ENSURE(context, lhs_data != nullptr);
  const TfLiteTensor* rhs_data = op_context.rhs;
  TF_LITE_ENSURE(context, rhs_data != nullptr);
  TfLiteTensor* output = op_context.output;
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, lhs_data->type == kTfLiteFloat32 ||
                              lhs_data->type == kTfLiteInt8 ||
                              lhs_data->type == kTfLiteInt16);
  TF_LITE_ENSURE(context, rhs_data->type == kTfLiteFloat32 ||
                              rhs_data->type == kTfLiteInt8 ||
                              rhs_data->type == kTfLiteInt16);
  // Both inputs should be of the same type.
  // Hybrid input (FLOAT32 LHS, INT8 RHS) is not supported.
  TF_LITE_ENSURE(context, lhs_data->type == rhs_data->type);
  // LHS input must match output type.  INT32 output not supported.
  TF_LITE_ENSURE(context, lhs_data->type == output->type);

  const int lhs_rank = NumDimensions(lhs_data);
  const int rhs_rank = NumDimensions(rhs_data);
  // Support dimensions between 2 and 5, inclusive.
  TF_LITE_ENSURE(context, lhs_rank >= 2);
  TF_LITE_ENSURE(context, lhs_rank <= 5);
  TF_LITE_ENSURE(context, rhs_rank >= 2);
  TF_LITE_ENSURE(context, rhs_rank <= 5);

  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, op_context));

  OpDataBatchMatmul* op_data = op_context.op_data;
  // If the RHS is constant, we only transpose once.
  op_data->rhs_is_transposed = false;
  op_data->lhs_is_constant_tensor = IsConstantTensor(lhs_data);
  op_data->rhs_is_constant_tensor = IsConstantTensor(rhs_data);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (lhs_data->type == kTfLiteInt8 || lhs_data->type == kTfLiteInt16) {
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, lhs_data, rhs_data, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier,
                       &op_data->quantization->output_multiplier,
                       &op_data->quantization->output_shift);
    // BatchMatMul has no fused activation functions. Therefore, set
    // output activation min and max to min and max of int8_t or int16_t type.
    if (lhs_data->type == kTfLiteInt8) {
      op_data->quantization->output_activation_min =
          std::numeric_limits<int8_t>::min();
      op_data->quantization->output_activation_max =
          std::numeric_limits<int8_t>::max();
    } else {
      op_data->quantization->output_activation_min =
          std::numeric_limits<int16_t>::min();
      op_data->quantization->output_activation_max =
          std::numeric_limits<int16_t>::max();

      TF_LITE_ENSURE_EQ(context, lhs_data->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, rhs_data->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    }

    op_data->quantization->lhs_zero_point = lhs_data->params.zero_point;
    op_data->quantization->rhs_zero_point = rhs_data->params.zero_point;
    op_data->quantization->output_zero_point = output->params.zero_point;
  }

  const int output_rank = std::max(lhs_rank, rhs_rank);
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(lhs_data));
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(rhs_data));

  // Ensure any batch dimensions obey broacasting rules.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    if (lhs_dim != rhs_dim) {
      if (lhs_dim != 1) {
        TF_LITE_ENSURE_EQ(context, rhs_dim, 1);
      }
    }
  }
  bool adj_x = op_context.params->adj_x;
  bool adj_y = op_context.params->adj_y;
  // Ensure other dimensions work for matrix multiplication.
  int accum_dim_lhs = adj_x ? extended_lhs_shape.Dims(output_rank - 2)
                            : extended_lhs_shape.Dims(output_rank - 1);
  int accum_dim_rhs = adj_y ? extended_rhs_shape.Dims(output_rank - 1)
                            : extended_rhs_shape.Dims(output_rank - 2);

  TF_LITE_ENSURE_EQ(context, accum_dim_lhs, accum_dim_rhs);
  TfLiteStatus status =
      ReshapeOutputTensor(context, node, extended_lhs_shape, extended_rhs_shape,
                          adj_x, adj_y, output_rank, output);
  return status;
}

TfLiteStatus EvalInt8(TfLiteContext* context, const OpDataBatchMatmul& data,
                      const RuntimeShape& lhs_shape,
                      const TfLiteEvalTensor& lhs,
                      const RuntimeShape& rhs_shape,
                      const TfLiteEvalTensor& rhs,
                      const RuntimeShape& output_shape,
                      TfLiteEvalTensor* output) {
  TF_LITE_ENSURE(context, data.quantization != nullptr);
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  op_params.input_offset = -data.quantization->lhs_zero_point;
  op_params.weights_offset =
      -data.quantization->rhs_zero_point;  // filter offset
  op_params.output_offset = data.quantization->output_zero_point;
  op_params.output_multiplier = data.quantization->output_multiplier;
  op_params.output_shift = data.quantization->output_shift;
  op_params.quantized_activation_min = data.quantization->output_activation_min;
  op_params.quantized_activation_max = data.quantization->output_activation_max;
  op_params.lhs_cacheable = data.lhs_is_constant_tensor;
  op_params.rhs_cacheable = data.rhs_is_constant_tensor;

  // Note we pass RHS args first, LHS args second. See note for Eval.
  reference_ops::BatchMatMul<int8_t, int32_t>(
      op_params, rhs_shape, tflite::micro::GetTensorData<int8_t>(&rhs),
      lhs_shape, tflite::micro::GetTensorData<int8_t>(&lhs), output_shape,
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus EvalInt16(TfLiteContext* context, const OpDataBatchMatmul& data,
                       const RuntimeShape& lhs_shape,
                       const TfLiteEvalTensor& lhs,
                       const RuntimeShape& rhs_shape,
                       const TfLiteEvalTensor& rhs,
                       const RuntimeShape& output_shape,
                       TfLiteEvalTensor* output) {
  TF_LITE_ENSURE(context, data.quantization != nullptr);
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  op_params.input_offset = -data.quantization->lhs_zero_point;
  op_params.weights_offset =
      -data.quantization->rhs_zero_point;  // filter offset
  op_params.output_offset = data.quantization->output_zero_point;
  op_params.output_multiplier = data.quantization->output_multiplier;
  op_params.output_shift = data.quantization->output_shift;
  op_params.quantized_activation_min = data.quantization->output_activation_min;
  op_params.quantized_activation_max = data.quantization->output_activation_max;
  op_params.lhs_cacheable = data.lhs_is_constant_tensor;
  op_params.rhs_cacheable = data.rhs_is_constant_tensor;

  // Note we pass RHS args first, LHS args second. See note for Eval.
  reference_ops::BatchMatMul<int16_t, int64_t>(
      op_params, rhs_shape, tflite::micro::GetTensorData<int16_t>(&rhs),
      lhs_shape, tflite::micro::GetTensorData<int16_t>(&lhs), output_shape,
      tflite::micro::GetTensorData<int16_t>(output));

  return kTfLiteOk;
}

// Perform a batch matrix multiply on
// LHS <..., A, B>  X  RHS<..., B, C>
// where the leading dimensions of LHS and RHS obey broadcasting rules
// (this Op will apply broadcasting rules).
// We assume that LHS and RHS are both row oriented (adjacent values in memory
// are in the same row) and will output in the same memory layout. However,
// our fast GEMM libraries assume RCC layout (LHS row oriented,
// RHS column oriented, output column oriented). Therefore, we perform
// RHS <..., C, B> X LHS <..., B, A>
// where output is a C X A column-oriented, which is equivalent to
// A X C row-oriented.
TfLiteStatus BatchMatMulEval(TfLiteContext* context, TfLiteNode* node) {
  EvalOpContext op_context(context, node);
  OpDataBatchMatmul* op_data = op_context.op_data;
  const TfLiteEvalTensor* lhs = op_context.lhs;
  const TfLiteEvalTensor* rhs = op_context.rhs;
  TfLiteEvalTensor* output = op_context.output;
  RuntimeShape orig_lhs_shape = tflite::micro::GetTensorShape(lhs);
  RuntimeShape orig_rhs_shape = tflite::micro::GetTensorShape(rhs);

  bool adj_y = op_context.params->adj_y;
  bool adj_x = op_context.params->adj_x;

  // Compress BatchMatMul when third from last RHS dimension is one.
  int32_t rhs_dims_count = orig_rhs_shape.DimensionsCount();
  int32_t lhs_dims_count = orig_lhs_shape.DimensionsCount();
  // Compress ops where rhs shape is [..., 1, X, Y] and lhs shape is
  // [..., Q, R, S] which is equivalent to rhs: [..., X, Y] and
  // lhs: [..., Q * R, S].
  if (rhs_dims_count > 2 && lhs_dims_count > 2) {
    int rhs_one = orig_rhs_shape.DimsData()[rhs_dims_count - 3];
    if (rhs_one == 1) {
      int32_t* lhs_dims = orig_lhs_shape.DimsData();
      int32_t* rhs_dims = orig_rhs_shape.DimsData();
      RuntimeShape tmp_l(lhs_dims_count - 1, lhs_dims);
      tmp_l.SetDim(lhs_dims_count - 3,
                   lhs_dims[lhs_dims_count - 3] * lhs_dims[lhs_dims_count - 2]);
      tmp_l.SetDim(lhs_dims_count - 2, lhs_dims[lhs_dims_count - 1]);
      orig_lhs_shape.ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
      RuntimeShape tmp_r(rhs_dims_count - 1, orig_rhs_shape.DimsData());
      tmp_r.SetDim(rhs_dims_count - 3, rhs_dims[rhs_dims_count - 2]);
      tmp_r.SetDim(rhs_dims_count - 2, rhs_dims[rhs_dims_count - 1]);
      orig_rhs_shape.ReplaceWith(tmp_r.DimensionsCount(), tmp_r.DimsData());
      rhs_dims_count = orig_rhs_shape.DimensionsCount();
      lhs_dims_count = orig_lhs_shape.DimensionsCount();
    }
  }

  TfLiteEvalTensor* rhs_tensor = adj_y ? const_cast<TfLiteEvalTensor*>(rhs)
                                       : op_data->rhs_transposed_tensor;
  TfLiteEvalTensor* lhs_tensor = adj_x ? op_data->lhs_transposed_tensor
                                       : const_cast<TfLiteEvalTensor*>(lhs);
  TF_LITE_ENSURE(context, rhs_tensor != nullptr);
  TF_LITE_ENSURE(context, lhs_tensor != nullptr);
  if (!adj_y) {
    // TODO(b/154760341): Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(op_data->rhs_is_constant_tensor && op_data->rhs_is_transposed)) {
      TransposeRowsColumns(*rhs, rhs_tensor);
      op_data->rhs_is_transposed = true;
    }
  }
  if (adj_x) {
    TransposeRowsColumns(*lhs, lhs_tensor);
  }
  RuntimeShape rhs_shape =
      adj_y ? orig_rhs_shape : SwapRowColumnDims(orig_rhs_shape);
  RuntimeShape lhs_shape =
      adj_x ? orig_lhs_shape : SwapRowColumnDims(orig_lhs_shape);

  switch (lhs->type) {
    case kTfLiteFloat32:
      // Note we pass RHS args first, LHS args second. See note above.
      reference_ops::BatchMatMul(
          rhs_shape, tflite::micro::GetTensorData<float>(rhs_tensor), lhs_shape,
          tflite::micro::GetTensorData<float>(lhs_tensor),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      return EvalInt8(context, *op_data, lhs_shape, *lhs_tensor, rhs_shape,
                      *rhs_tensor, tflite::micro::GetTensorShape(output),
                      output);
    case kTfLiteInt16:
      return EvalInt16(context, *op_data, lhs_shape, *lhs_tensor, rhs_shape,
                       *rhs_tensor, tflite::micro::GetTensorShape(output),
                       output);
    default:
      MicroPrintf("BATCH_MATMUL doesn't support input type %s",
                  TfLiteTypeGetName(lhs->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_BATCH_MATMUL() {
  return tflite::micro::RegisterOp(BatchMatMulInit, BatchMatMulPrepare,
                                   BatchMatMulEval);
}

}  // namespace tflite
