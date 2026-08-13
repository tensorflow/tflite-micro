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

#include "tensorflow/lite/kernels/internal/reference/maximum_minimum.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/maximum_minimum.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {

namespace {

#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
// This file has the HiFi implementation of TFMaximum/TFMinimum.
enum KernelType {
  kHiFi5,
  kReference,
};

namespace hifi {

enum class basic_op {
  no_op,
  min,
  max
};

template <typename data_type>
TfLiteStatus TFLiteOperation( TfLiteContext* context, TfLiteNode* node,
                              const OpContext& op_context,
                              basic_op operation);

template <typename T, int NDims>
int MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const T* input1_data,
                             const RuntimeShape& unextended_input2_shape, const T* input2_data,
                             const RuntimeShape& unextended_output_shape,       T* output_data,
                             hifi::basic_op op);

template <typename data_type>
int ExecElemKernel( hifi::basic_op op, data_type *out_data,
                    const data_type *in1_data, const data_type *in2_data, size_t num_Elements);

}  // namespace hifi

/* Execute an element-wise kernel depending on the type specified by elem_op.
 * For now, the only data_type supported is int8_t
 * hifi::basic_op::min -> xa_nn_elm_min_8x8_8
 * hifi::basic_op::max -> xa_nn_elm_max_8x8_8
 */
template <typename data_type>
int hifi::ExecElemKernel( hifi::basic_op elem_op, data_type *out_data,
                          const data_type *in1_data, const data_type *in2_data,
                          size_t num_Elements) {
  int err = 0;
  
  if(!std::is_same<data_type, int8_t>::value){
      err = -1;
  } else {
    switch(elem_op){
      case hifi::basic_op::min :
        err = xa_nn_elm_min_8x8_8(out_data, in1_data, in2_data, num_Elements);
        break;
      case hifi::basic_op::max :
        err = xa_nn_elm_max_8x8_8(out_data, in1_data, in2_data, num_Elements);
        break;
      default :
        err = -1;
    }
  }

  return err;
}

/*
 * Invoke element-wise kernels if the shapes match, else,
 * call hifi::MaximumMinimumBroadcast(...)
 */
template <typename data_type>
TfLiteStatus hifi::TFLiteOperation(
    TfLiteContext* context, TfLiteNode* node,
    const OpContext& op_context, hifi::basic_op elem_op) {

  int err = 0;

  const RuntimeShape &unextended_input1_shape = tflite::micro::GetTensorShape(op_context.input1),
                     &unextended_input2_shape = tflite::micro::GetTensorShape(op_context.input2),
                     &unextended_output_shape = tflite::micro::GetTensorShape(op_context.output);

  if (unextended_input1_shape == unextended_input2_shape) {
    err = hifi::ExecElemKernel<data_type>( elem_op,
                    tflite::micro::GetTensorData<data_type>(op_context.output),
                    tflite::micro::GetTensorData<data_type>(op_context.input1),
                    tflite::micro::GetTensorData<data_type>(op_context.input2),
                    unextended_output_shape.FlatSize());
  } else {
    int maxDims = std::max( { unextended_input1_shape.DimensionsCount(),
                              unextended_input2_shape.DimensionsCount(),
                              unextended_output_shape.DimensionsCount()} );

    /* Verify that number of dimensions is between 1 to 8 */
    TFLITE_DCHECK_GE(maxDims, 1);
    TFLITE_DCHECK_LE(maxDims, 8);

    //decltype(&hifi::MaximumMinimumBroadcast<data_type, 4>) bCastMinMax = NULL;

    if (maxDims <= 4) {
      err = hifi::MaximumMinimumBroadcast<data_type, 4>(
              tflite::micro::GetTensorShape(op_context.input1), tflite::micro::GetTensorData<data_type>(op_context.input1),
              tflite::micro::GetTensorShape(op_context.input2), tflite::micro::GetTensorData<data_type>(op_context.input2),
              tflite::micro::GetTensorShape(op_context.output), tflite::micro::GetTensorData<data_type>(op_context.output),
              elem_op);
    } else {
      err = hifi::MaximumMinimumBroadcast<data_type, 8>(
              tflite::micro::GetTensorShape(op_context.input1), tflite::micro::GetTensorData<data_type>(op_context.input1),
              tflite::micro::GetTensorShape(op_context.input2), tflite::micro::GetTensorData<data_type>(op_context.input2),
              tflite::micro::GetTensorShape(op_context.output), tflite::micro::GetTensorData<data_type>(op_context.output),
              elem_op);
    }
  }

  TF_LITE_ENSURE_EQ(context, err, 0);

  return kTfLiteOk;
}

template <typename data_type, int NDims>
int hifi::MaximumMinimumBroadcast( const RuntimeShape& unextended_input1_shape, const data_type *input1_data,
                                   const RuntimeShape& unextended_input2_shape, const data_type *input2_data,
                                   const RuntimeShape& unextended_output_shape,       data_type *output_data,
                                   hifi::basic_op op) {
  int err = 0;

  TFLITE_DCHECK_GE(unextended_input1_shape.DimensionsCount(), 0);
  TFLITE_DCHECK_GE(unextended_input2_shape.DimensionsCount(), 0);
  TFLITE_DCHECK_GE(unextended_output_shape.DimensionsCount(), 0);

  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), NDims);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), NDims);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), NDims);

  const RuntimeShape extended_input1_shape = RuntimeShape::ExtendedShape(NDims, unextended_input1_shape);
  const RuntimeShape extended_input2_shape = RuntimeShape::ExtendedShape(NDims, unextended_input2_shape);
  const RuntimeShape extended_output_shape = RuntimeShape::ExtendedShape(NDims, unextended_output_shape);

  NdArrayDesc<NDims> input1_desc;
  NdArrayDesc<NDims> input2_desc;
  NdArrayDesc<NDims> output_desc;

  NdArrayDescsForElementwiseBroadcast( unextended_input1_shape, unextended_input2_shape,
                                                 &input1_desc,            &input2_desc );

  CopyDimsToDesc(extended_output_shape, &output_desc);

  if ( !(extended_input1_shape == extended_output_shape) &&              // input 1 needs broadcast
       (extended_input2_shape == extended_output_shape)     ) {

    err = xa_nn_broadcast_8_8(output_data, output_desc.extents,                       
            input1_data, extended_input1_shape.DimsData(), NDims);      // broadcast input 1 into output_data buffer

    err |= hifi::ExecElemKernel(op, output_data,                        // exec element-wise op after bcast
                   output_data, input2_data,
                   extended_output_shape.FlatSize());

  } else if( (extended_input1_shape == extended_output_shape) &&
             !(extended_input2_shape == extended_output_shape)     ) {   // input 2 needs broadcast

    err = xa_nn_broadcast_8_8(output_data, output_desc.extents,                        
            input2_data, extended_input2_shape.DimsData(), NDims);      // broadcast input 2 into output_data buffer

    err |= hifi::ExecElemKernel(op, output_data,                        // exec element-wise op after bcast
                   input1_data, output_data,
                   extended_output_shape.FlatSize());
  } 
#if (defined(HIFI4) || defined(HIFI5))
  else {

    /* Both inputs need broadcast.
     * Call any of the xa_nn_elm_[min,max]_[4D,8D]_Bcast_8x8_8(...) kernels.
     * All these kernels have same the same function signature.
     */

    decltype(&xa_nn_elm_min_4D_Bcast_8x8_8) kernel = NULL;

    if (NDims == 4) {
      kernel = (op == hifi::basic_op::min) ? xa_nn_elm_min_4D_Bcast_8x8_8 :
                 (op == hifi::basic_op::max) ? xa_nn_elm_max_4D_Bcast_8x8_8 : NULL;
    } else if (NDims == 8) {
      kernel = (op == hifi::basic_op::min) ? xa_nn_elm_min_8D_Bcast_8x8_8 :
                 (op == hifi::basic_op::max) ? xa_nn_elm_max_8D_Bcast_8x8_8 : NULL;
    }

    if(kernel!=NULL){
      err = kernel(output_data, output_desc.extents,
                   input1_data, input1_desc.strides,
                   input2_data, input2_desc.strides );
    }
  }
#endif

  return err;
}

#endif // defined(HIFI5) || defined(HIFI4)

template <KernelType kernel_type, typename OpType>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);

#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
  if (kernel_type == kReference || kernel_type == kHiFi5) {

    hifi::basic_op operation =
        std::is_same<OpType, MinimumOp>::value ? hifi::basic_op::min :
            std::is_same<OpType, MaximumOp>::value ? hifi::basic_op::max :
                hifi::basic_op::no_op;
#else
  if (kernel_type == kReference) {
#endif

    switch (op_context.output->type) {
      case kTfLiteFloat32:
        TFLiteOperation<float, OpType>(context, node, op_context);
        break;
      case kTfLiteInt8:
#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
        hifi::TFLiteOperation<int8_t>(context, node, op_context, operation);
#else
        TFLiteOperation<int8_t, OpType>(context, node, op_context);
#endif
        break;
      case kTfLiteInt16:
        TFLiteOperation<int16_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt32:
        TFLiteOperation<int32_t, OpType>(context, node, op_context);
        break;
      case kTfLiteInt64:
        TFLiteOperation<int64_t, OpType>(context, node, op_context);
        break;
      default:
        MicroPrintf("Type %s (%d) is not supported by Maximum/Minimum.",
                    TfLiteTypeGetName(op_context.output->type),
                    op_context.output->type);
        return kTfLiteError;
    }
  } else {
    MicroPrintf("Kernel type not supported by Maximum/Minimum.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_MAXIMUM() {
  return tflite::micro::RegisterOp(
      nullptr, nullptr,
#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
          Eval<kHiFi5, MaximumOp>
#else
          Eval<kReference, MaximumOp>
#endif
  );                                
}

TFLMRegistration Register_MINIMUM() {
  return tflite::micro::RegisterOp(
      nullptr, nullptr,
#if (defined(HIFI4) || defined(HIFI5) || defined(HIFI_IQ))
          Eval<kHiFi5, MinimumOp>
#else
          Eval<kReference, MinimumOp>
#endif
  );                                
}

}  // namespace tflite
