/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#define TFLM_COMPILE_LIBRARY

#include "python/tflite_micro/runtime_c_api.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "flatbuffers/util.h"
#include "python/tflite_micro/compression_utils.h"
#include "python/tflite_micro/python_ops_resolver.h"
#include "python/tflite_micro/shared_library.h"
#include "tensorflow/lite/micro/flatbuffer_conversions.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

struct TFLM_Interpreter {
  std::unique_ptr<uint8_t[]> memory_arena;
  std::unique_ptr<uint8_t[]> alt_decompression_memory;
  tflite::MicroContext::AlternateMemoryRegion alt_decompression_region;
  tflite::PythonOpsResolver python_ops_resolver;
  tflite::MicroAllocator* allocator = nullptr;
  tflite::RecordingMicroAllocator* recording_allocator = nullptr;
  std::unique_ptr<tflite::MicroInterpreter> interpreter;
  std::string last_error;
};

namespace {

void SetError(char* err_buf, size_t err_buf_len, const char* msg) {
  if (err_buf != nullptr && err_buf_len > 0) {
    std::snprintf(err_buf, err_buf_len, "%s", msg);
  }
}

TFLM_Status Fail(TFLM_Interpreter* interp, std::string msg) {
  if (interp != nullptr) {
    interp->last_error = std::move(msg);
  }
  return kTFLMError;
}

int TfLiteTypeToSchemaType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return tflite::TensorType_FLOAT32;
    case kTfLiteFloat16:
      return tflite::TensorType_FLOAT16;
    case kTfLiteInt32:
      return tflite::TensorType_INT32;
    case kTfLiteUInt8:
      return tflite::TensorType_UINT8;
    case kTfLiteInt64:
      return tflite::TensorType_INT64;
    case kTfLiteString:
      return tflite::TensorType_STRING;
    case kTfLiteBool:
      return tflite::TensorType_BOOL;
    case kTfLiteInt16:
      return tflite::TensorType_INT16;
    case kTfLiteComplex64:
      return tflite::TensorType_COMPLEX64;
    case kTfLiteInt8:
      return tflite::TensorType_INT8;
    case kTfLiteFloat64:
      return tflite::TensorType_FLOAT64;
    case kTfLiteComplex128:
      return tflite::TensorType_COMPLEX128;
    case kTfLiteUInt64:
      return tflite::TensorType_UINT64;
    case kTfLiteResource:
      return tflite::TensorType_RESOURCE;
    case kTfLiteVariant:
      return tflite::TensorType_VARIANT;
    case kTfLiteUInt32:
      return tflite::TensorType_UINT32;
    case kTfLiteUInt16:
      return tflite::TensorType_UINT16;
    case kTfLiteInt4:
      return tflite::TensorType_INT4;
    case kTfLiteBFloat16:
      return tflite::TensorType_BFLOAT16;
    case kTfLiteInt2:
      return tflite::TensorType_INT2;
    case kTfLiteUInt4:
      return tflite::TensorType_UINT4;
    default:
      return -1;
  }
}

bool AddCustomOpRegistererByName(const char* registerer_name,
                                 tflite::PythonOpsResolver* resolver) {
  typedef bool (*RegistererFunctionType)(tflite::PythonOpsResolver*);
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
      tflite::SharedLibrary::GetSymbol(registerer_name));

  if (registerer == nullptr) {
    MicroPrintf("Looking up symbol '%s' failed with error '%s'.",
                registerer_name, tflite::SharedLibrary::GetError());
    return false;
  }

  if (!registerer(resolver)) {
    MicroPrintf("%s failed to register op.", registerer_name);
    return false;
  }
  return true;
}

}  // namespace

extern "C" {

TFLM_Interpreter* TFLM_InterpreterCreate(
    const uint8_t* model_data, size_t model_size,
    const char* const* custom_op_registerers, size_t num_registerers,
    size_t arena_size, int num_resource_variables, int interpreter_config,
    size_t alt_decompression_memory_size, char* err_buf, size_t err_buf_len) {
  if (model_data == nullptr || model_size == 0) {
    SetError(err_buf, err_buf_len, "Invalid model data.");
    return nullptr;
  }

  const tflite::Model* model = tflite::GetModel(model_data);
  if (model == nullptr) {
    SetError(err_buf, err_buf_len, "Failed to parse model flatbuffer.");
    return nullptr;
  }

  if (!tflite::IsCompressionSupported() &&
      tflite::HasCompressionMetadata(*model)) {
    SetError(err_buf, err_buf_len,
             "Model contains compressed tensors but the interpreter was not "
             "built with compression support. Please build with "
             "--//:with_compression=true.");
    return nullptr;
  }

  auto wrapper = std::make_unique<TFLM_Interpreter>();
  wrapper->memory_arena = std::make_unique<uint8_t[]>(arena_size);
  if (alt_decompression_memory_size > 0) {
    wrapper->alt_decompression_memory =
        std::make_unique<uint8_t[]>(alt_decompression_memory_size);
    wrapper->alt_decompression_region = {
        wrapper->alt_decompression_memory.get(), alt_decompression_memory_size};
  } else {
    wrapper->alt_decompression_region = {nullptr, 0};
  }

  if (custom_op_registerers != nullptr) {
    for (size_t i = 0; i < num_registerers; ++i) {
      if (!AddCustomOpRegistererByName(custom_op_registerers[i],
                                       &wrapper->python_ops_resolver)) {
        std::string err =
            std::string("TFLM could not register custom op via ") +
            custom_op_registerers[i];
        SetError(err_buf, err_buf_len, err.c_str());
        return nullptr;
      }
    }
  }

  switch (interpreter_config) {
    case kTFLMConfigAllocationRecording: {
      wrapper->recording_allocator = tflite::RecordingMicroAllocator::Create(
          wrapper->memory_arena.get(), arena_size);
      wrapper->allocator = wrapper->recording_allocator;
      break;
    }
    case kTFLMConfigPreserveAllTensors: {
      wrapper->allocator = tflite::MicroAllocator::Create(
          wrapper->memory_arena.get(), arena_size,
          tflite::MemoryPlannerType::kLinear);
      break;
    }
    default:
      SetError(err_buf, err_buf_len, "Unknown interpreter config.");
      return nullptr;
  }

  tflite::MicroResourceVariables* resource_variables = nullptr;
  if (num_resource_variables > 0) {
    resource_variables = tflite::MicroResourceVariables::Create(
        wrapper->allocator, num_resource_variables);
  }

  wrapper->interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, wrapper->python_ops_resolver, wrapper->allocator,
      resource_variables);

  if (alt_decompression_memory_size > 0) {
    TfLiteStatus status = wrapper->interpreter->SetDecompressionMemory(
        &wrapper->alt_decompression_region, 1);
    if (status != kTfLiteOk) {
      SetError(err_buf, err_buf_len,
               "TFLM failed to set decompression memory.");
      return nullptr;
    }
  }

  TfLiteStatus status = wrapper->interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    SetError(err_buf, err_buf_len, "TFLM failed to allocate tensors.");
    return nullptr;
  }

  return wrapper.release();
}

void TFLM_InterpreterDestroy(TFLM_Interpreter* interp) { delete interp; }

const char* TFLM_InterpreterLastError(const TFLM_Interpreter* interp) {
  if (interp == nullptr) {
    return "Null interpreter instance.";
  }
  return interp->last_error.c_str();
}

TFLM_Status TFLM_InterpreterInvoke(TFLM_Interpreter* interp) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    return Fail(interp, "Null interpreter instance.");
  }

  if (interp->interpreter->Invoke() != kTfLiteOk) {
    return Fail(interp, "Interpreter invocation failed.");
  }
  return kTFLMOk;
}

TFLM_Status TFLM_InterpreterReset(TFLM_Interpreter* interp) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    return Fail(interp, "Null interpreter instance.");
  }
  if (interp->interpreter->Reset() != kTfLiteOk) {
    return Fail(interp, "Failed to reset interpreter.");
  }
  return kTFLMOk;
}

void TFLM_InterpreterPrintAllocations(TFLM_Interpreter* interp) {
  if (interp != nullptr && interp->recording_allocator != nullptr) {
    interp->recording_allocator->PrintAllocations();
  }
}

TFLM_Status TFLM_InterpreterSetInputData(TFLM_Interpreter* interp, size_t index,
                                         const void* data, size_t byte_size,
                                         int tensor_type, const int* dims,
                                         size_t num_dims) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    return Fail(interp, "Null interpreter instance.");
  }

  TfLiteTensor* tensor = interp->interpreter->input(index);
  if (tensor == nullptr) {
    return Fail(interp, "Tensor is out of bounds, please check tensor index.");
  }

  if (tensor_type != -1) {
    TfLiteType input_type = kTfLiteNoType;
    if (tflite::ConvertTensorType(static_cast<tflite::TensorType>(tensor_type),
                                  &input_type) != kTfLiteOk ||
        tensor->type != input_type) {
      char msg[128];
      std::snprintf(msg, sizeof(msg),
                    "Cannot set tensor: Got value of type %s but expected type "
                    "%s for input %zu",
                    TfLiteTypeGetName(input_type),
                    TfLiteTypeGetName(tensor->type), index);
      return Fail(interp, msg);
    }
  }

  if (dims != nullptr) {
    if (num_dims != static_cast<size_t>(tensor->dims->size)) {
      char msg[128];
      std::snprintf(msg, sizeof(msg),
                    "Cannot set tensor: Dimension mismatch. Got %zu but "
                    "expected %d for input %zu",
                    num_dims, tensor->dims->size, index);
      return Fail(interp, msg);
    }
    for (size_t j = 0; j < num_dims; ++j) {
      if (tensor->dims->data[j] != dims[j]) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "Cannot set tensor: Dimension mismatch. Got %d but "
                      "expected %d for dimension %zu of input %zu",
                      dims[j], tensor->dims->data[j], j, index);
        return Fail(interp, msg);
      }
    }
  }

  if (tensor->bytes != byte_size) {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "numpy array had %zu bytes but expected %zu bytes.",
                  byte_size, tensor->bytes);
    return Fail(interp, msg);
  }

  if (tensor->data.data == nullptr && tensor->bytes > 0) {
    return Fail(interp,
                "Cannot set tensor: Tensor is non-empty but has nullptr.");
  }

  std::memcpy(tensor->data.data, data, byte_size);
  return kTFLMOk;
}

TFLM_Status TFLM_InterpreterGetTensor(TFLM_Interpreter* interp, int kind,
                                      size_t index, size_t subgraph_index,
                                      TFLM_TensorInfo* out_info) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    return Fail(interp, "Null interpreter instance.");
  }
  if (out_info == nullptr) {
    return Fail(interp, "Null output tensor info.");
  }
  *out_info = {};

  if (kind == kTFLMTensorEval) {
    if (!interp->interpreter->preserve_all_tensors()) {
      return Fail(
          interp,
          "TFLM only supports GetTensor() when using a python interpreter "
          "with the InterpreterConfig.kPreserveAllTensors interpreter_config");
    }

    const TfLiteEvalTensor* eval_tensor =
        interp->interpreter->GetTensor(index, subgraph_index);
    if (eval_tensor == nullptr) {
      return Fail(interp, "Tensor not found.");
    }

    out_info->tensor_type = TfLiteTypeToSchemaType(eval_tensor->type);
    out_info->data = eval_tensor->data.data;
    out_info->bytes = tflite::EvalTensorBytes(eval_tensor);
    if (eval_tensor->dims != nullptr) {
      out_info->dims = eval_tensor->dims->data;
      out_info->num_dims = static_cast<size_t>(eval_tensor->dims->size);
    }
    return kTFLMOk;
  }

  const TfLiteTensor* tensor = (kind == kTFLMTensorInput)
                                   ? interp->interpreter->input(index)
                                   : interp->interpreter->output(index);
  if (tensor == nullptr) {
    return Fail(interp, "Tensor is out of bound, please check tensor index.");
  }

  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    return Fail(
        interp,
        "TFLM doesn't support strings, resource variables, or variants as "
        "outputs.");
  }

  out_info->tensor_type = TfLiteTypeToSchemaType(tensor->type);
  out_info->data = tensor->data.data;
  out_info->bytes = tensor->bytes;
  if (tensor->dims != nullptr) {
    out_info->dims = tensor->dims->data;
    out_info->num_dims = static_cast<size_t>(tensor->dims->size);
  }

  if (tensor->quantization.type == kTfLiteAffineQuantization &&
      tensor->quantization.params != nullptr) {
    const auto* q_params = reinterpret_cast<const TfLiteAffineQuantization*>(
        tensor->quantization.params);
    if (q_params->scale != nullptr && q_params->zero_point != nullptr) {
      static_assert(sizeof(int) == sizeof(int32_t),
                    "TfLiteIntArray element size must match int32_t");
      out_info->scales = q_params->scale->data;
      out_info->zero_points =
          reinterpret_cast<const int32_t*>(q_params->zero_point->data);
      out_info->num_quant_params = static_cast<size_t>(q_params->scale->size);
    }
    out_info->quantized_dimension = q_params->quantized_dimension;
  }

  return kTFLMOk;
}

TFLM_Status TFLM_AlignFlatbuffer(const char* input_path,
                                 const char* output_path, char* err_buf,
                                 size_t err_buf_len) {
  if (input_path == nullptr || output_path == nullptr) {
    SetError(err_buf, err_buf_len, "Null path provided.");
    return kTFLMError;
  }

  std::string model_file;
  if (!flatbuffers::LoadFile(input_path, false, &model_file)) {
    SetError(err_buf, err_buf_len, "Failed to load input flatbuffer file.");
    return kTFLMError;
  }

  const tflite::Model* model = tflite::GetModel(model_file.c_str());
  if (model == nullptr) {
    SetError(err_buf, err_buf_len, "Invalid tflite model.");
    return kTFLMError;
  }

  tflite::ModelT* unpacked_model = model->UnPack();
  flatbuffers::FlatBufferBuilder fbb;
  auto new_model = tflite::Model::Pack(fbb, unpacked_model);
  delete unpacked_model;

  fbb.Finish(new_model, tflite::ModelIdentifier());
  if (!flatbuffers::SaveFile(output_path,
                             reinterpret_cast<char*>(fbb.GetBufferPointer()),
                             fbb.GetSize(), /*binary=*/true)) {
    SetError(err_buf, err_buf_len, "Failed to write aligned flatbuffer file.");
    return kTFLMError;
  }

  return kTFLMOk;
}

float TFLM_FreqToMel(float freq) { return 1127.0f * log1pf(freq / 700.0f); }

namespace {
constexpr size_t kWideDynamicFunctionBits = 32;
constexpr size_t kWideDynamicFunctionLUTSize =
    (4 * kWideDynamicFunctionBits - 3);

int16_t PcanGainLookupFunction(const float strength, const float offset,
                               const int gain_bits, int32_t input_bits,
                               uint32_t x) {
  const float x_as_float =
      static_cast<float>(x) / (static_cast<uint32_t>(1) << input_bits);
  const float gain_as_float = (static_cast<uint32_t>(1) << gain_bits) *
                              powf(x_as_float + offset, -strength);

  if (gain_as_float > std::numeric_limits<int16_t>::max()) {
    return std::numeric_limits<int16_t>::max();
  }
  return static_cast<int16_t>(std::lround(gain_as_float));
}
}  // namespace

TFLM_Status TFLM_WideDynamicFuncLut(float strength, float offset,
                                    int input_bits, int gain_bits,
                                    int16_t* out_lut, size_t max_lut_size,
                                    size_t* out_lut_size) {
  if (out_lut == nullptr || max_lut_size < kWideDynamicFunctionLUTSize) {
    return kTFLMError;
  }

  int16_t gain_lut[kWideDynamicFunctionLUTSize + 1];
  gain_lut[0] =
      PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 0);
  gain_lut[1] =
      PcanGainLookupFunction(strength, offset, gain_bits, input_bits, 1);

  for (size_t interval = 2; interval <= kWideDynamicFunctionBits; ++interval) {
    const uint32_t x0 = static_cast<uint32_t>(1) << (interval - 1);
    const uint32_t x1 = x0 + (x0 >> 1);
    const uint32_t x2 =
        (interval == kWideDynamicFunctionBits) ? x0 + (x0 - 1) : 2 * x0;

    const int16_t y0 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x0);
    const int16_t y1 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x1);
    const int16_t y2 =
        PcanGainLookupFunction(strength, offset, gain_bits, input_bits, x2);

    const int32_t diff1 = static_cast<int32_t>(y1 - y0);
    const int32_t diff2 = static_cast<int32_t>(y2 - y0);
    const int32_t a1 = 4 * diff1 - diff2;
    const int32_t a2 = diff2 - a1;

    const size_t base_idx = 4 * interval - 6;
    gain_lut[base_idx] = y0;
    gain_lut[base_idx + 1] = static_cast<int16_t>(a1);
    gain_lut[base_idx + 2] = static_cast<int16_t>(a2);
    gain_lut[base_idx + 3] = 0;
  }

  for (size_t i = 0; i < kWideDynamicFunctionLUTSize; ++i) {
    out_lut[i] = gain_lut[i];
  }
  if (out_lut_size != nullptr) {
    *out_lut_size = kWideDynamicFunctionLUTSize;
  }
  return kTFLMOk;
}

}  // extern "C"
