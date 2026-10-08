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
#include "tensorflow/lite/micro/flatbuffer_conversions.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "flatbuffers/vector.h"  // from @flatbuffers
#include "tensorflow/lite/micro/builtin_op_data.h"
#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace micro {

namespace {

// Utility class for safely allocating POD data. This is useful for avoiding
// leaks in cases where op params are allocated but fail to propagate to the
// parsed op data (e.g., when model parameters are invalid).
class SafeBuiltinDataAllocator {
 public:
  class BuiltinDataDeleter {
   public:
    explicit BuiltinDataDeleter(BuiltinDataAllocator* allocator)
        : allocator_(allocator) {}

    void operator()(void* data) { allocator_->Deallocate(data); }

   private:
    BuiltinDataAllocator* allocator_;
  };

  template <typename T>
  using BuiltinDataPtr = std::unique_ptr<T, BuiltinDataDeleter>;

  explicit SafeBuiltinDataAllocator(BuiltinDataAllocator* allocator)
      : allocator_(allocator) {}

  template <typename T>
  BuiltinDataPtr<T> Allocate() {
    return BuiltinDataPtr<T>(allocator_->AllocatePOD<T>(),
                             BuiltinDataDeleter(allocator_));
  }

 private:
  BuiltinDataAllocator* allocator_;
};

// All the Parse functions take some pointers as params and this function has
// the common DCHECKs to catch if any of those are nullptr.
void CheckParsePointerParams(const Operator* op,
                             BuiltinDataAllocator* allocator,
                             void** builtin_data) {
  TFLITE_DCHECK(op != nullptr);
  TFLITE_DCHECK(allocator != nullptr);
  TFLITE_DCHECK(builtin_data != nullptr);
}

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
template <typename DataType = int32_t>
static TfLiteStatus FlatBufferIntVectorToArray(
    int max_size_of_buffer, const flatbuffers::Vector<DataType>* flat_vector,
    DataType* buffer, const char* op_name) {
  if (!flat_vector) {
    MicroPrintf("Input array not provided for operation '%s'.\n", op_name);
    return kTfLiteError;
  } else {
    size_t num_dimensions = flat_vector->size();
    if (num_dimensions > max_size_of_buffer / sizeof(DataType)) {
      MicroPrintf(
          "Found too many dimensions in the input array of operation '%s'.\n",
          op_name);
      return kTfLiteError;
    } else {
      for (size_t i = 0; i < num_dimensions; ++i) {
        buffer[i] = flat_vector->Get(i);
      }
    }
  }
  return kTfLiteOk;
}

// Converts the flatbuffer activation to what is used at runtime.
TfLiteFusedActivation ConvertActivation(ActivationFunctionType activation) {
  switch (activation) {
    case ActivationFunctionType_NONE:
      return kTfLiteActNone;
    case ActivationFunctionType_RELU:
      return kTfLiteActRelu;
    case ActivationFunctionType_RELU_N1_TO_1:
      return kTfLiteActReluN1To1;
    case ActivationFunctionType_RELU6:
      return kTfLiteActRelu6;
    case ActivationFunctionType_TANH:
      return kTfLiteActTanh;
    case ActivationFunctionType_SIGN_BIT:
      return kTfLiteActSignBit;
  }
  return kTfLiteActNone;
}

// Converts the flatbuffer padding enum to what is used at runtime.
TfLitePadding ConvertPadding(Padding padding) {
  switch (padding) {
    case Padding_SAME:
      return kTfLitePaddingSame;
    case Padding_VALID:
      return kTfLitePaddingValid;
  }
  return kTfLitePaddingUnknown;
}

// Converts the flatbuffer mirror padding enum to what is used at runtime.
TfLiteMirrorPaddingMode ConvertMirrorPadding(MirrorPadMode padding) {
  switch (padding) {
    case MirrorPadMode_REFLECT:
      return kTfLiteMirrorPaddingReflect;
    case MirrorPadMode_SYMMETRIC:
      return kTfLiteMirrorPaddingSymmetric;
  }
  return kTfLiteMirrorPaddingUnknown;
}

TfLiteRngAlgorithm ConvertRngAlgorithm(RngAlgorithm algorithm) {
  switch (algorithm) {
    case RngAlgorithm_THREEFRY:
      return kTfLiteRngAlgorithmThreefry;
    case RngAlgorithm_PHILOX:
      return kTfLiteRngAlgorithmPhilox;
    case RngAlgorithm_DEFAULT:
      return kTfLiteRngAlgorithmDefault;
  }
  return kTfLiteRngAlgorithmUnknown;
}

}  // namespace

TfLiteStatus ConvertTensorType(TensorType tensor_type, TfLiteType* type) {
  switch (tensor_type) {
    case TensorType_FLOAT16:
      *type = kTfLiteFloat16;
      return kTfLiteOk;
    case TensorType_BFLOAT16:
      *type = kTfLiteBFloat16;
      return kTfLiteOk;
    case TensorType_FLOAT32:
      *type = kTfLiteFloat32;
      return kTfLiteOk;
    case TensorType_FLOAT64:
      *type = kTfLiteFloat64;
      return kTfLiteOk;
    case TensorType_INT16:
      *type = kTfLiteInt16;
      return kTfLiteOk;
    case TensorType_UINT16:
      *type = kTfLiteUInt16;
      return kTfLiteOk;
    case TensorType_INT32:
      *type = kTfLiteInt32;
      return kTfLiteOk;
    case TensorType_UINT32:
      *type = kTfLiteUInt32;
      return kTfLiteOk;
    case TensorType_UINT8:
      *type = kTfLiteUInt8;
      return kTfLiteOk;
    case TensorType_INT8:
      *type = kTfLiteInt8;
      return kTfLiteOk;
    case TensorType_INT64:
      *type = kTfLiteInt64;
      return kTfLiteOk;
    case TensorType_UINT64:
      *type = kTfLiteUInt64;
      return kTfLiteOk;
    case TensorType_STRING:
      *type = kTfLiteString;
      return kTfLiteOk;
    case TensorType_BOOL:
      *type = kTfLiteBool;
      return kTfLiteOk;
    case TensorType_COMPLEX64:
      *type = kTfLiteComplex64;
      return kTfLiteOk;
    case TensorType_COMPLEX128:
      *type = kTfLiteComplex128;
      return kTfLiteOk;
    case TensorType_RESOURCE:
      *type = kTfLiteResource;
      return kTfLiteOk;
    case TensorType_VARIANT:
      *type = kTfLiteVariant;
      return kTfLiteOk;
    case TensorType_INT4:
      *type = kTfLiteInt4;
      return kTfLiteOk;
    case TensorType_INT2:
      *type = kTfLiteInt2;
      return kTfLiteOk;
    case TensorType_UINT4:
      *type = kTfLiteUInt4;
      return kTfLiteOk;
    case TensorType_FLOAT8_E4M3FN:
      *type = kTfLiteFloat8E4M3FN;
      return kTfLiteOk;
    case TensorType_FLOAT8_E5M2:
      *type = kTfLiteFloat8E5M2;
      return kTfLiteOk;
    default:
      *type = kTfLiteNoType;
      MicroPrintf("Unsupported data type %d in tensor\n", tensor_type);
      return kTfLiteError;
  }
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseAbs(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseAdd(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteAddParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteAddParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const AddOptions* schema_params = op->builtin_options_as_AddOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseAddN(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  return kTfLiteOk;
}

TfLiteStatus ParseArgMax(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMaxParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ArgMaxOptions* schema_params = op->builtin_options_as_ArgMaxOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(
        ConvertTensorType(schema_params->output_type(), &params->output_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseArgMin(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteArgMinParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteArgMinParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ArgMinOptions* schema_params = op->builtin_options_as_ArgMinOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(
        ConvertTensorType(schema_params->output_type(), &params->output_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseAssignVariable(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBatchMatMul(const Operator* op,
                              BuiltinDataAllocator* allocator,
                              void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteBatchMatMulParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* bmm_params = op->builtin_options_as_BatchMatMulOptions()) {
    params->adj_x = bmm_params->adj_x();
    params->adj_y = bmm_params->adj_y();
    params->asymmetric_quantize_inputs =
        bmm_params->asymmetric_quantize_inputs();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBatchToSpaceNd(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBroadcastArgs(const Operator*, BuiltinDataAllocator*,
                                void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBroadcastTo(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseCallOnce(const Operator* op, BuiltinDataAllocator* allocator,
                           void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteCallOnceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteCallOnceParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const CallOnceOptions* schema_params =
      op->builtin_options_as_CallOnceOptions();

  if (schema_params != nullptr) {
    params->init_subgraph_index = schema_params->init_subgraph_index();

  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCast(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCastParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* schema_params = op->builtin_options_as_CastOptions()) {
    TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->in_data_type(),
                                            &params->in_data_type));
    TF_LITE_ENSURE_STATUS(ConvertTensorType(schema_params->out_data_type(),
                                            &params->out_data_type));
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCeil(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseConcatenation(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConcatenationParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConcatenationParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ConcatenationOptions* schema_params =
      op->builtin_options_as_ConcatenationOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseConv2D(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteConvParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const Conv2DOptions* schema_params = op->builtin_options_as_Conv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
    TF_LITE_ENSURE_STATUS(ConvertTensorType(
        schema_params->quantized_bias_type(), &params->quantized_bias_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCumsum(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteCumsumParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* cumsum_params = op->builtin_options_as_CumsumOptions()) {
    params->exclusive = cumsum_params->exclusive();
    params->reverse = cumsum_params->reverse();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseCos(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseDepthToSpace(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteDepthToSpaceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthToSpaceParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const auto* schema_params = op->builtin_options_as_DepthToSpaceOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const DepthwiseConv2DOptions* schema_params =
      op->builtin_options_as_DepthwiseConv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->depth_multiplier = schema_params->depth_multiplier();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseDequantize(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseDiv(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteDivParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* schema_params = op->builtin_options_as_DivOptions()) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseDynamicUpdateSlice(const Operator*, BuiltinDataAllocator*,
                                     void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseElu(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseEmbeddingLookup(const Operator*, BuiltinDataAllocator*,
                                  void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseExp(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseExpandDims(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFill(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloor(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloorDiv(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseFloorMod(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseFullyConnected(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteFullyConnectedParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteFullyConnectedParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const FullyConnectedOptions* schema_params =
      op->builtin_options_as_FullyConnectedOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->keep_num_dims = schema_params->keep_num_dims();
    params->asymmetric_quantize_inputs =
        schema_params->asymmetric_quantize_inputs();
    TF_LITE_ENSURE_STATUS(ConvertTensorType(
        schema_params->quantized_bias_type(), &params->quantized_bias_type));
    switch (schema_params->weights_format()) {
      case FullyConnectedOptionsWeightsFormat_DEFAULT:
        params->weights_format = kTfLiteFullyConnectedWeightsFormatDefault;
        break;
      case FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8:
        params->weights_format =
            kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8;
        break;
      default:
        MicroPrintf("Unhandled fully-connected weights format.");
        return kTfLiteError;
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGather(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteGatherParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  params->axis = 0;
  params->batch_dims = 0;
  if (const auto* gather_params = op->builtin_options_as_GatherOptions()) {
    params->axis = gather_params->axis();
    params->batch_dims = gather_params->batch_dims();
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGatherNd(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGreater(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseGreaterEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseHardSwish(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseIf(const Operator* op, BuiltinDataAllocator* allocator,
                     void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteIfParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteIfParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const IfOptions* schema_params = op->builtin_options_as_IfOptions();

  if (schema_params != nullptr) {
    params->then_subgraph_index = schema_params->then_subgraph_index();
    params->else_subgraph_index = schema_params->else_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseL2Normalization(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteL2NormParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteL2NormParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const L2NormOptions* schema_params = op->builtin_options_as_L2NormOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseLeakyRelu(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteLeakyReluParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* leaky_relu_params =
          op->builtin_options_as_LeakyReluOptions()) {
    params->alpha = leaky_relu_params->alpha();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLess(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLessEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLog(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalAnd(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalNot(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogicalOr(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogistic(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseLogSoftmax(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseLSTM(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteLSTMParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* lstm_params = op->builtin_options_as_LSTMOptions()) {
    params->activation =
        ConvertActivation(lstm_params->fused_activation_function());
    params->cell_clip = lstm_params->cell_clip();
    params->proj_clip = lstm_params->proj_clip();
    switch (lstm_params->kernel_type()) {
      case LSTMKernelType_FULL:
        params->kernel_type = kTfLiteLSTMFullKernel;
        break;
      case LSTMKernelType_BASIC:
        params->kernel_type = kTfLiteLSTMBasicKernel;
        break;
      default:
        MicroPrintf("Unhandled LSTM kernel type: %d",
                    lstm_params->kernel_type());
        return kTfLiteError;
    }
    params->asymmetric_quantize_inputs =
        lstm_params->asymmetric_quantize_inputs();
  } else {
    MicroPrintf("No valid LSTM builtin options exist");
    return kTfLiteError;
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseMaximum(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseMinimum(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseMirrorPad(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMirrorPaddingParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMirrorPaddingParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const MirrorPadOptions* schema_params =
      op->builtin_options_as_MirrorPadOptions();

  if (schema_params != nullptr) {
    params->mode = ConvertMirrorPadding(schema_params->mode());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseMul(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteMulParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteMulParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const MulOptions* schema_params = op->builtin_options_as_MulOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // Default activation is none.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseNeg(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseNotEqual(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParsePack(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePackParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const PackOptions* schema_params = op->builtin_options_as_PackOptions();

  if (schema_params != nullptr) {
    params->values_count = schema_params->values_count();
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePad(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePadV2(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParsePool(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLitePoolParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLitePoolParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const Pool2DOptions* schema_params = op->builtin_options_as_Pool2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->filter_width = schema_params->filter_width();
    params->filter_height = schema_params->filter_height();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePow(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParsePrelu(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseQuantize(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseReadVariable(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseReducer(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReducerParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReducerParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ReducerOptions* schema_params = op->builtin_options_as_ReducerOptions();

  if (schema_params != nullptr) {
    params->keep_dims = schema_params->keep_dims();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRelu(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRelu6(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseReshape(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteReshapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteReshapeParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ReshapeOptions* schema_params = op->builtin_options_as_ReshapeOptions();

  if (schema_params != nullptr) {
    const flatbuffers::Vector<int32_t>* new_shape = schema_params->new_shape();
    if (new_shape != nullptr) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray(
          sizeof(params->shape), new_shape, params->shape, "reshape"));
      params->num_dimensions = new_shape->size();
    } else {
      // TODO(b/157480169) TODO(b/147203660): We should either return
      // kTfLiteError or fill in some reasonable defaults in the params struct.
      // We are not doing so until we better undertand the ramifications of
      // changing the legacy behavior.
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseResizeBilinear(const Operator* op,
                                 BuiltinDataAllocator* allocator,
                                 void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeBilinearParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeBilinearParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ResizeBilinearOptions* schema_params =
      op->builtin_options_as_ResizeBilinearOptions();

  if (schema_params != nullptr) {
    params->align_corners = schema_params->align_corners();
    params->half_pixel_centers = schema_params->half_pixel_centers();
  } else {
    params->align_corners = false;
    params->half_pixel_centers = false;
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseResizeNearestNeighbor(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteResizeNearestNeighborParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteResizeNearestNeighborParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ResizeNearestNeighborOptions* schema_params =
      op->builtin_options_as_ResizeNearestNeighborOptions();

  if (schema_params != nullptr) {
    params->align_corners = schema_params->align_corners();
    params->half_pixel_centers = schema_params->half_pixel_centers();
  } else {
    params->align_corners = false;
    params->half_pixel_centers = false;
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseStablehloReduceWindow(const Operator* op,
                                        BuiltinDataAllocator* allocator,
                                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloReduceWindowParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StablehloReduceWindowOptions* schema_params =
      op->builtin_options_2_as_StablehloReduceWindowOptions();
  if (schema_params) {
    if (!schema_params->window_dimensions() ||
        schema_params->window_dimensions()->size() == 0) {
      MicroPrintf(
          "'window_dimensions' attribute is not optional for "
          "'stablehlo.reduce_window' and cannot be empty.");
      return kTfLiteError;
    }

    const size_t rank = schema_params->window_dimensions()->size();

    auto LoadAttr = [](int64_t* params_array, size_t params_array_size_bytes,
                       const flatbuffers::Vector<int64_t>* flatbuffer_vector,
                       const char* attr_name, const size_t expected_size,
                       const int64_t fill_value) -> TfLiteStatus {
      if (flatbuffer_vector && flatbuffer_vector->size()) {
        if (expected_size != 0 && flatbuffer_vector->size() != expected_size) {
          MicroPrintf(
              "'%s' attribute of 'stablehlo.reduce_window' does not have the "
              "expected size (%llu != %llu).",
              attr_name, flatbuffer_vector->size(), expected_size);
          return kTfLiteError;
        }
        TfLiteStatus status = FlatBufferIntVectorToArray(
            params_array_size_bytes, flatbuffer_vector, params_array,
            "stablehlo.reduce_window");
        if (status != kTfLiteOk) {
          MicroPrintf("Check the '%s' attribute.", attr_name);
          return status;
        }
      } else {
        std::fill_n(params_array, params_array_size_bytes / sizeof(int64_t),
                    fill_value);
      }
      return kTfLiteOk;
    };

    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->window_dimensions, sizeof(params->window_dimensions),
                 schema_params->window_dimensions(), "window_dimensions",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->window_strides, sizeof(params->window_strides),
                 schema_params->window_strides(), "window_strides",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->base_dilations, sizeof(params->base_dilations),
                 schema_params->base_dilations(), "base_dilations",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->window_dilations, sizeof(params->window_dilations),
                 schema_params->window_dilations(), "window_dilations",
                 /*expected_size=*/rank, /*fill_value=*/1));
    TF_LITE_ENSURE_STATUS(LoadAttr(params->padding, sizeof(params->padding),
                                   schema_params->padding(), "padding",
                                   /*expected_size=*/2 * rank,
                                   /*fill_value=*/0));

    params->body_subgraph_index = schema_params->body_subgraph_index();
    *builtin_data = params.release();
    return kTfLiteOk;
  }
  MicroPrintf("Could not get 'stablehlo.reduce_window' operation parameters.");
  return kTfLiteError;
}

TfLiteStatus ParseStablehloScatter(const Operator* op,
                                   BuiltinDataAllocator* allocator,
                                   void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloScatterParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloScatterParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StablehloScatterOptions* schema_params =
      op->builtin_options_2_as_StablehloScatterOptions();
  if (schema_params) {
    params->indices_are_sorted = schema_params->indices_are_sorted();

    if (schema_params->update_window_dims()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->update_window_dims()->size() * sizeof(int64_t),
          schema_params->update_window_dims(), params->update_window_dims,
          "stablehlo_scatter"));
      params->num_update_window_dims =
          schema_params->update_window_dims()->size();
    }

    if (schema_params->inserted_window_dims()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->inserted_window_dims()->size() * sizeof(int64_t),
          schema_params->inserted_window_dims(), params->inserted_window_dims,
          "stablehlo_scatter"));
      params->num_inserted_window_dims =
          schema_params->inserted_window_dims()->size();
    }

    if (schema_params->scatter_dims_to_operand_dims()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->scatter_dims_to_operand_dims()->size() *
              sizeof(int64_t),
          schema_params->scatter_dims_to_operand_dims(),
          params->scatter_dims_to_operand_dims, "stablehlo_scatter"));
      params->num_scatter_dims_to_operand_dims =
          schema_params->scatter_dims_to_operand_dims()->size();
    }

    params->index_vector_dim = schema_params->index_vector_dim();
    params->unique_indices = schema_params->unique_indices();
    params->update_computation_subgraph_index =
        schema_params->update_computation_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseStablehloRngBitGenerator(const Operator* op,
                                           BuiltinDataAllocator* allocator,
                                           void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloRngBitGeneratorParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloRngBitGeneratorParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StablehloRngBitGeneratorOptions* schema_params =
      op->builtin_options_2_as_StablehloRngBitGeneratorOptions();
  if (schema_params != nullptr) {
    params->algorithm = ConvertRngAlgorithm(schema_params->algorithm());
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseStablehloGather(const Operator* op,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStablehloGatherParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStablehloGatherParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StablehloGatherOptions* schema_params =
      op->builtin_options_2_as_StablehloGatherOptions();

  if (schema_params != nullptr) {
    if (schema_params->offset_dims()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          /*max_size_of_buffer=*/schema_params->offset_dims()->size() *
              sizeof(int64_t),
          /*flat_vector=*/schema_params->offset_dims(),
          /*buffer=*/params->offset_dims, /*op_name=*/"stablehlo_gather"));
      params->num_offset_dims = schema_params->offset_dims()->size();
    }

    if (schema_params->collapsed_slice_dims()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->collapsed_slice_dims()->size() * sizeof(int64_t),
          schema_params->collapsed_slice_dims(), params->collapsed_slice_dims,
          "stablehlo_gather"));
      params->num_collapsed_slice_dims =
          schema_params->collapsed_slice_dims()->size();
    }

    if (schema_params->start_index_map()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->start_index_map()->size() * sizeof(int64_t),
          schema_params->start_index_map(), params->start_index_map,
          "stablehlo_gather"));
      params->num_start_index_map = schema_params->start_index_map()->size();
    }

    params->index_vector_dim = schema_params->index_vector_dim();

    if (schema_params->slice_sizes()) {
      TF_LITE_ENSURE_STATUS(FlatBufferIntVectorToArray<int64_t>(
          schema_params->slice_sizes()->size() * sizeof(int64_t),
          schema_params->slice_sizes(), params->slice_sizes,
          "stablehlo_gather"));
      params->num_slice_sizes = schema_params->slice_sizes()->size();
    }

    params->indices_are_sorted = schema_params->indices_are_sorted();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseStablehloPad(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloPadParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  const StablehloPadOptions* schema_params =
      op->builtin_options_2_as_StablehloPadOptions();

  if (schema_params) {
    auto LoadAttr =
        [](int64_t* params_array, const size_t params_array_size_bytes,
           const flatbuffers::Vector<int64_t>* const flatbuffer_vector,
           const char* const attr_name) -> TfLiteStatus {
      TfLiteStatus status =
          FlatBufferIntVectorToArray(params_array_size_bytes, flatbuffer_vector,
                                     params_array, "stablehlo.pad");
      if (status != kTfLiteOk) {
        MicroPrintf("Check the '%s' attribute.", attr_name);
      }
      return status;
    };

    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->edge_padding_low, sizeof(params->edge_padding_low),
                 schema_params->edge_padding_low(), "edge_padding_low"));
    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->edge_padding_high, sizeof(params->edge_padding_high),
                 schema_params->edge_padding_high(), "edge_padding_high"));
    TF_LITE_ENSURE_STATUS(
        LoadAttr(params->interior_padding, sizeof(params->interior_padding),
                 schema_params->interior_padding(), "interior_padding"));
    if (schema_params->edge_padding_low()->size() !=
            schema_params->edge_padding_high()->size() ||
        schema_params->edge_padding_low()->size() !=
            schema_params->interior_padding()->size()) {
      MicroPrintf(
          "'stablehlo.pad' operation parameter array sizes "
          "are not consistent.");
      return kTfLiteError;
    }
    *builtin_data = params.release();
    return kTfLiteOk;
  }
  MicroPrintf("Could not get 'stablehlo.pad' operation parameters.");
  return kTfLiteError;
}

TfLiteStatus ParseStablehloComposite(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloCompositeParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  const StableHLOCompositeOptions* schema_params =
      op->builtin_options_2_as_StableHLOCompositeOptions();
  if (schema_params) {
    if (schema_params->name() == nullptr) {
      MicroPrintf("'stablehlo.composite' missing required option 'name'.");
      return kTfLiteError;
    }
    if (schema_params->composite_attributes() == nullptr) {
      MicroPrintf(
          "'stablehlo.composite' missing required option "
          "'composite_attributes'.");
      return kTfLiteError;
    }
    params->name = schema_params->name()->c_str();
    params->version = schema_params->version();
    params->subgraph_index = schema_params->decomposition_subgraph_index();
    params->attributes = schema_params->composite_attributes()->data();
    params->attributes_size = schema_params->composite_attributes()->size();
    *builtin_data = params.release();
    return kTfLiteOk;
  }
  MicroPrintf("Could not get 'stablehlo.composite' operation parameters.");
  return kTfLiteError;
}

TfLiteStatus ParseStablehloShiftLeft(const Operator* op,
                                     BuiltinDataAllocator* allocator,
                                     void** builtin_data) {
  return kTfLiteOk;
}

TfLiteStatus ParseStablehloCase(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params = safe_allocator.Allocate<TfLiteStablehloCaseParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StablehloCaseOptions* schema_params =
      op->builtin_options_2_as_StablehloCaseOptions();
  if (schema_params) {
    auto LoadAttr =
        [](int32_t* params_array, const size_t params_array_size_bytes,
           const flatbuffers::Vector<int32_t>* const flatbuffer_vector,
           const char* const attr_name) -> TfLiteStatus {
      TfLiteStatus status =
          FlatBufferIntVectorToArray(params_array_size_bytes, flatbuffer_vector,
                                     params_array, "stablehlo.case");
      if (status != kTfLiteOk) {
        MicroPrintf("Check the '%s' attribute.", attr_name);
      }
      return status;
    };

    TF_LITE_ENSURE_STATUS(LoadAttr(params->branch_subgraph_indices,
                                   sizeof(params->branch_subgraph_indices),
                                   schema_params->branch_subgraph_indices(),
                                   "branch subgraph indices"));
    params->num_branches = schema_params->branch_subgraph_indices()->size();
    *builtin_data = params.release();
    return kTfLiteOk;
  }
  MicroPrintf("Could not get 'stablehlo.case' operation parameters.");
  return kTfLiteError;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseReverseV2(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRound(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRsqrt(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSelect(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSelectV2(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseShape(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteShapeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteShapeParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const ShapeOptions* schema_params = op->builtin_options_as_ShapeOptions();

  if (schema_params != nullptr) {
    TF_LITE_ENSURE_STATUS(
        ConvertTensorType(schema_params->out_type(), &params->out_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSin(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSlice(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseSoftmax(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSoftmaxParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSoftmaxParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SoftmaxOptions* schema_params = op->builtin_options_as_SoftmaxOptions();

  if (schema_params != nullptr) {
    params->beta = schema_params->beta();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSpaceToBatchNd(const Operator*, BuiltinDataAllocator*,
                                 void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseSpaceToDepth(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSpaceToDepthParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSpaceToDepthParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const auto* schema_params = op->builtin_options_as_SpaceToDepthOptions();
  if (schema_params != nullptr) {
    params->block_size = schema_params->block_size();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSplit(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSplitParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SplitOptions* schema_params = op->builtin_options_as_SplitOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSplitV(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSplitVParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSplitVParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SplitVOptions* schema_params = op->builtin_options_as_SplitVOptions();

  if (schema_params != nullptr) {
    params->num_splits = schema_params->num_splits();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseUnidirectionalSequenceLSTM(const Operator* op,
                                             BuiltinDataAllocator* allocator,
                                             void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);
  auto params =
      safe_allocator.Allocate<TfLiteUnidirectionalSequenceLSTMParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  if (const auto* seq_lstm_params =
          op->builtin_options_as_UnidirectionalSequenceLSTMOptions()) {
    params->activation =
        ConvertActivation(seq_lstm_params->fused_activation_function());
    params->cell_clip = seq_lstm_params->cell_clip();
    params->proj_clip = seq_lstm_params->proj_clip();
    params->time_major = seq_lstm_params->time_major();
    params->asymmetric_quantize_inputs =
        seq_lstm_params->asymmetric_quantize_inputs();
    params->diagonal_recurrent_tensors =
        seq_lstm_params->diagonal_recurrent_tensors();
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSqueeze(const Operator* op, BuiltinDataAllocator* allocator,
                          void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);
  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteSqueezeParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSqueezeParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SqueezeOptions* schema_params = op->builtin_options_as_SqueezeOptions();

  if (schema_params != nullptr) {
    const auto* squeeze_dims = schema_params->squeeze_dims();
    if (squeeze_dims != nullptr) {
      TF_LITE_ENSURE_STATUS(
          FlatBufferIntVectorToArray(sizeof(params->squeeze_dims), squeeze_dims,
                                     params->squeeze_dims, "squeeze"));
      params->num_squeeze_dims = squeeze_dims->size();
    } else {
      params->num_squeeze_dims = 0;
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSqrt(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSquare(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseSquaredDifference(const Operator*, BuiltinDataAllocator*,
                                    void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseStridedSlice(const Operator* op,
                               BuiltinDataAllocator* allocator,
                               void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteStridedSliceParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteStridedSliceParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const StridedSliceOptions* schema_params =
      op->builtin_options_as_StridedSliceOptions();

  if (schema_params != nullptr) {
    params->begin_mask = schema_params->begin_mask();
    params->end_mask = schema_params->end_mask();
    params->ellipsis_mask = schema_params->ellipsis_mask();
    params->new_axis_mask = schema_params->new_axis_mask();
    params->shrink_axis_mask = schema_params->shrink_axis_mask();
    params->offset = schema_params->offset();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSub(const Operator* op, BuiltinDataAllocator* allocator,
                      void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSubParams, SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSubParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SubOptions* schema_params = op->builtin_options_as_SubOptions();

  if (schema_params != nullptr) {
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->pot_scale_int16 = schema_params->pot_scale_int16();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseSvdf(const Operator* op, BuiltinDataAllocator* allocator,
                       void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteSVDFParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteSVDFParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const SVDFOptions* schema_params = op->builtin_options_as_SVDFOptions();
  if (schema_params != nullptr) {
    params->rank = schema_params->rank();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());
    params->asymmetric_quantize_inputs =
        schema_params->asymmetric_quantize_inputs();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseTanh(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}
//
// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseTranspose(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus ParseTransposeConv(const Operator* op,
                                BuiltinDataAllocator* allocator,
                                void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteTransposeConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteTransposeConvParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }
  const TransposeConvOptions* transpose_conv_params =
      op->builtin_options_as_TransposeConvOptions();
  if (transpose_conv_params != nullptr) {
    params->padding = ConvertPadding(transpose_conv_params->padding());
    params->stride_width = transpose_conv_params->stride_w();
    params->stride_height = transpose_conv_params->stride_h();

    params->activation =
        ConvertActivation(transpose_conv_params->fused_activation_function());
    TF_LITE_ENSURE_STATUS(
        ConvertTensorType(transpose_conv_params->quantized_bias_type(),
                          &params->quantized_bias_type));
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }
  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseUnpack(const Operator* op, BuiltinDataAllocator* allocator,
                         void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteUnpackParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteUnpackParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const UnpackOptions* schema_params = op->builtin_options_as_UnpackOptions();

  if (schema_params != nullptr) {
    params->num = schema_params->num();
    params->axis = schema_params->axis();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseVarHandle(const Operator* op, BuiltinDataAllocator* allocator,
                            void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteVarHandleParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteVarHandleParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const VarHandleOptions* schema_params =
      op->builtin_options_as_VarHandleOptions();

  if (schema_params != nullptr) {
    if (schema_params->container()) {
      params->container = schema_params->container()->c_str();
    }
    if (schema_params->shared_name()) {
      params->shared_name = schema_params->shared_name()->c_str();
    }
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

TfLiteStatus ParseWhile(const Operator* op, BuiltinDataAllocator* allocator,
                        void** builtin_data) {
  CheckParsePointerParams(op, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);
  std::unique_ptr<TfLiteWhileParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteWhileParams>();
  if (params == nullptr) {
    MicroPrintf("Failed to allocate memory for builtin_data\n");
    return kTfLiteError;
  }

  const WhileOptions* schema_params = op->builtin_options_as_WhileOptions();

  if (schema_params != nullptr) {
    params->cond_subgraph_index = schema_params->cond_subgraph_index();
    params->body_subgraph_index = schema_params->body_subgraph_index();
  } else {
    // TODO(b/157480169): We should either return kTfLiteError or fill in some
    // reasonable defaults in the params struct. We are not doing so until we
    // better understand the ramifications of changing the legacy behavior.
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseZerosLike(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseBitwiseXor(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

// We have this parse function instead of directly returning kTfLiteOk from the
// switch-case in ParseOpData because this function is used as part of the
// selective registration for the OpResolver implementation in micro.
TfLiteStatus ParseRightShift(const Operator*, BuiltinDataAllocator*, void**) {
  return kTfLiteOk;
}

TfLiteStatus CallBuiltinParseFunction(TfLiteBridgeBuiltinParseFunction parser,
                                      const Operator* op,
                                      BuiltinDataAllocator* allocator,
                                      void** builtin_data) {
  return parser(op, allocator, builtin_data);
}

}  // namespace micro
}  // namespace tflite
