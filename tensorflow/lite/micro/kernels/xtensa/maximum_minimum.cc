/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace ops {
namespace micro {
namespace maximum_minimum {
namespace {

// This file has the HiFi implementation of TFMaximum/TFMinimum.
enum KernelType {
  kHiFi,
  kReference,
};

constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input1 = tflite::micro::GetEvalInput(context, node, kInputTensor1);
    input2 = tflite::micro::GetEvalInput(context, node, kInputTensor2);
    output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  }
  const TfLiteEvalTensor* input1;
  const TfLiteEvalTensor* input2;
  TfLiteEvalTensor* output;
};

struct MaximumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 > el2 ? el1 : el2;
  }
};

struct MinimumOp {
  template <typename data_type>
  static data_type op(data_type el1, data_type el2) {
    return el1 < el2 ? el1 : el2;
  }
};

}  // namespace maximum_minimum

#if defined(HIFI5)
namespace hifi {

template <typename data_type, typename op_type>
TfLiteStatus TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context);

template <typename T, typename Op, int N = 4>
void MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const T* input1_data,
                                        const RuntimeShape& unextended_input2_shape, const T* input2_data,
                                        const RuntimeShape& unextended_output_shape,       T* output_data,
                                        Op op);
}  // namespace hifi
#endif // defined(HIFI5)

template <typename data_type, typename op_type>
void TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context) {
  reference_ops::MaximumMinimumBroadcastSlow(
      tflite::micro::GetTensorShape(op_context.input1),
      tflite::micro::GetTensorData<data_type>(op_context.input1),
      tflite::micro::GetTensorShape(op_context.input2),
      tflite::micro::GetTensorData<data_type>(op_context.input2),
      tflite::micro::GetTensorShape(op_context.output),
      tflite::micro::GetTensorData<data_type>(op_context.output),
      op_type::template op<data_type>);
}

#if defined(HIFI5)
template <typename data_type, typename op_type, int N>
void hifi::MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const data_type *input1_data,
                                    const RuntimeShape& unextended_input2_shape, const data_type *input2_data,
                                    const RuntimeShape& unextended_output_shape,       data_type *output_data,
                                    op_type op) {

    TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), N);
    TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), N);
    TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), N);

    NdArrayDesc<N> input1_desc;
    NdArrayDesc<N> input2_desc;
    NdArrayDesc<N> output_desc;

    NdArrayDescsForElementwiseBroadcast(
        unextended_input1_shape, unextended_input2_shape, &input1_desc, &input2_desc);

    CopyDimsToDesc(RuntimeShape::ExtendedShape(N, unextended_output_shape),
                   &output_desc);

    if(std::is_same<op_type, MaximumOp>::value) {
      xa_nn_elm_max_4D_Bcast_8x8_8( output_data, output_desc.extents,
                                    input1_data, input1_desc.strides,
                                    input2_data, input2_desc.strides);
    }

    if(std::is_same<op_type, MinimumOp>::value) {
      xa_nn_elm_min_4D_Bcast_8x8_8( output_data, output_desc.extents,
                                    input1_data, input1_desc.strides,
                                    input2_data, input2_desc.strides);

    }
}


template <typename data_type, typename op_type>
TfLiteStatus hifi::TFLiteOperation(TfLiteContext* context, TfLiteNode* node,
                     const OpContext& op_context) {
  
  const RuntimeShape &unextended_input1_shape = tflite::micro::GetTensorShape(op_context.input1),
                     &unextended_input2_shape = tflite::micro::GetTensorShape(op_context.input2),
                     &unextended_output_shape = tflite::micro::GetTensorShape(op_context.output);

  int err = 0;
  

  if(unextended_input1_shape == unextended_input2_shape){
    if(std::is_same<op_type, MaximumOp>::value) {
      xa_nn_elm_max_8x8_8( tflite::micro::GetTensorData<data_type>(op_context.output),
                           tflite::micro::GetTensorData<data_type>(op_context.input1),
                           tflite::micro::GetTensorData<data_type>(op_context.input2),
                           unextended_output_shape.FlatSize());
      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }
    if(std::is_same<op_type, MinimumOp>::value) {
      xa_nn_elm_min_8x8_8( tflite::micro::GetTensorData<data_type>(op_context.output),
                           tflite::micro::GetTensorData<data_type>(op_context.input1),
                           tflite::micro::GetTensorData<data_type>(op_context.input2),
                           unextended_output_shape.FlatSize());
      TF_LITE_MICRO_EXPECT_EQ(0, err);
    }
  } else {
    op_type maxmin_type;
    hifi::MaximumMinimumBroadcast(
        tflite::micro::GetTensorShape(op_context.input1), tflite::micro::GetTensorData<data_type>(op_context.input1),
        tflite::micro::GetTensorShape(op_context.input2), tflite::micro::GetTensorData<data_type>(op_context.input2),
        tflite::micro::GetTensorShape(op_context.output), tflite::micro::GetTensorData<data_type>(op_context.output),
        maxmin_type);
  }
  
  return kTfLiteOk;
}
#endif // defined(HIFI5)

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

#if defined(HIFI5)
  if ( (kernel_type == kHiFi) &&
      (op_context.output->type == kTfLiteInt8) )
  {
    switch (op_context.output->type) {
      case kTfLiteInt8:
        hifi::TFLiteOperation<int8_t, OpType>(context, node, op_context);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Type %s (%d) is not supported by Maximum/Minimum.",
                           TfLiteTypeGetName(op_context.output->type),
                           op_context.output->type);
        return kTfLiteError;
    }
  } else 
#endif // defined(HIFI5)
  if ((kernel_type == kReference)  ||
        ((kernel_type == kHiFi)))
  {

    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteUInt8:
        TFLiteOperation<uint8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt8:
        TFLiteOperation<int8_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt32:
        TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Type %s (%d) is not supported by Maximum/Minimum.",
                           TfLiteTypeGetName(op_context.output->type),
                           op_context.output->type);
        return kTfLiteError;
    }
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Kernel type not supported by Maximum/Minimum.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace maximum_minimum

TfLiteRegistration Register_MAXIMUM() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/
#if defined(HIFI5)
          maximum_minimum::Eval<maximum_minimum::kHiFi,
                                maximum_minimum::MaximumOp>,
#else
          maximum_minimum::Eval<maximum_minimum::kReference,
                                maximum_minimum::MaximumOp>,
#endif // defined(HIFI5)
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MINIMUM() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/nullptr,
          /*invoke=*/
#if defined(HIFI5)
          maximum_minimum::Eval<maximum_minimum::kHiFi,
                                maximum_minimum::MinimumOp>,
#else
          maximum_minimum::Eval<maximum_minimum::kReference,
                                maximum_minimum::MinimumOp>,
#endif // defined(HIFI5)
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
