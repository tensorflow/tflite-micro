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

#include "tensorflow/lite/micro/c_api/tflm_c_api.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers/util.h"
#include "python/tflite_micro/compression_utils.h"
#include "python/tflite_micro/python_ops_resolver.h"
#include "python/tflite_micro/shared_library.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {

void SetError(char* err_buf, size_t err_buf_len, const char* msg) {
  if (err_buf != nullptr && err_buf_len > 0) {
    std::snprintf(err_buf, err_buf_len, "%s", msg);
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

struct TFLM_Interpreter {
  std::unique_ptr<uint8_t[]> memory_arena;
  std::unique_ptr<uint8_t[]> alt_decompression_memory;
  tflite::MicroContext::AlternateMemoryRegion alt_decompression_region;
  tflite::PythonOpsResolver python_ops_resolver;
  tflite::MicroAllocator* allocator = nullptr;
  tflite::RecordingMicroAllocator* recording_allocator = nullptr;
  std::unique_ptr<tflite::MicroInterpreter> interpreter;
};

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

TFLM_Status TFLM_InterpreterInvoke(TFLM_Interpreter* interp, char* err_buf,
                                   size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return kTFLMError;
  }

  TfLiteStatus status = interp->interpreter->Invoke();
  if (status != kTfLiteOk) {
    SetError(err_buf, err_buf_len, "Interpreter invocation failed.");
    return kTFLMError;
  }
  return kTFLMOk;
}

TFLM_Status TFLM_InterpreterReset(TFLM_Interpreter* interp, char* err_buf,
                                  size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return kTFLMError;
  }
  if (interp->interpreter->Reset() != kTfLiteOk) {
    SetError(err_buf, err_buf_len, "Failed to reset interpreter.");
    return kTFLMError;
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
                                         int type_code, const int* dims,
                                         size_t num_dims, char* err_buf,
                                         size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return kTFLMError;
  }

  TfLiteTensor* tensor = interp->interpreter->input(index);
  if (tensor == nullptr) {
    SetError(err_buf, err_buf_len,
             "Tensor is out of bounds, please check tensor index.");
    return kTFLMError;
  }

  if (type_code != -1 && tensor->type != static_cast<TfLiteType>(type_code)) {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "Cannot set tensor: Got value of type %s but expected type "
                  "%s for input %zu",
                  TfLiteTypeGetName(static_cast<TfLiteType>(type_code)),
                  TfLiteTypeGetName(tensor->type), index);
    SetError(err_buf, err_buf_len, msg);
    return kTFLMError;
  }

  if (dims != nullptr) {
    if (num_dims != static_cast<size_t>(tensor->dims->size)) {
      char msg[128];
      std::snprintf(msg, sizeof(msg),
                    "Cannot set tensor: Dimension mismatch. Got %zu but "
                    "expected %d for input %zu",
                    num_dims, tensor->dims->size, index);
      SetError(err_buf, err_buf_len, msg);
      return kTFLMError;
    }
    for (size_t j = 0; j < num_dims; ++j) {
      if (tensor->dims->data[j] != dims[j]) {
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "Cannot set tensor: Dimension mismatch. Got %d but "
                      "expected %d for dimension %zu of input %zu",
                      dims[j], tensor->dims->data[j], j, index);
        SetError(err_buf, err_buf_len, msg);
        return kTFLMError;
      }
    }
  }

  if (tensor->bytes != byte_size) {
    char msg[128];
    std::snprintf(msg, sizeof(msg),
                  "numpy array had %zu bytes but expected %zu bytes.",
                  byte_size, tensor->bytes);
    SetError(err_buf, err_buf_len, msg);
    return kTFLMError;
  }

  if (tensor->data.data == nullptr && tensor->bytes > 0) {
    SetError(err_buf, err_buf_len,
             "Cannot set tensor: Tensor is non-empty but has nullptr.");
    return kTFLMError;
  }

  std::memcpy(tensor->data.data, data, byte_size);
  return kTFLMOk;
}

const void* TFLM_InterpreterGetOutputData(const TFLM_Interpreter* interp,
                                          size_t index, size_t* out_byte_size,
                                          char* err_buf, size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return nullptr;
  }

  const TfLiteTensor* tensor = interp->interpreter->output(index);
  if (tensor == nullptr) {
    SetError(err_buf, err_buf_len, "Output tensor index out of bounds.");
    return nullptr;
  }

  if (out_byte_size != nullptr) {
    *out_byte_size = tensor->bytes;
  }
  return tensor->data.data;
}

const void* TFLM_InterpreterGetTensorData(TFLM_Interpreter* interp,
                                          size_t tensor_index,
                                          size_t subgraph_index,
                                          size_t* out_byte_size, char* err_buf,
                                          size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return nullptr;
  }

  if (!interp->interpreter->preserve_all_tensors()) {
    SetError(
        err_buf, err_buf_len,
        "TFLM only supports GetTensor() when using a python interpreter "
        "with the InterpreterConfig.kPreserveAllTensors interpreter_config");
    return nullptr;
  }

  const TfLiteEvalTensor* eval_tensor =
      interp->interpreter->GetTensor(tensor_index, subgraph_index);
  if (eval_tensor == nullptr) {
    SetError(err_buf, err_buf_len, "Tensor not found.");
    return nullptr;
  }

  if (out_byte_size != nullptr) {
    *out_byte_size = tflite::EvalTensorBytes(eval_tensor);
  }
  return eval_tensor->data.data;
}

TFLM_Status TFLM_InterpreterGetTensorDetails(
    const TFLM_Interpreter* interp, int is_input, size_t index, int* out_type,
    int* out_dims, size_t* out_num_dims, size_t max_dims, float* out_scales,
    int32_t* out_zero_points, size_t* out_num_quant_params,
    size_t max_quant_params, int* out_quantized_dimension, char* err_buf,
    size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return kTFLMError;
  }

  const TfLiteTensor* tensor = is_input ? interp->interpreter->input(index)
                                        : interp->interpreter->output(index);
  if (tensor == nullptr) {
    SetError(err_buf, err_buf_len,
             "Tensor is out of bound, please check tensor index.");
    return kTFLMError;
  }

  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    SetError(err_buf, err_buf_len,
             "TFLM doesn't support strings, resource variables, or variants as "
             "outputs.");
    return kTFLMError;
  }

  if (out_type != nullptr) {
    *out_type = static_cast<int>(tensor->type);
  }

  if (out_dims != nullptr && out_num_dims != nullptr &&
      tensor->dims != nullptr) {
    size_t count = static_cast<size_t>(tensor->dims->size);
    if (count > max_dims) {
      count = max_dims;
    }
    *out_num_dims = count;
    for (size_t i = 0; i < count; ++i) {
      out_dims[i] = tensor->dims->data[i];
    }
  }

  const TfLiteQuantization quantization = tensor->quantization;
  size_t q_count = 0;
  int32_t q_dim = 0;
  if (quantization.type == kTfLiteAffineQuantization &&
      quantization.params != nullptr) {
    const auto* q_params =
        reinterpret_cast<const TfLiteAffineQuantization*>(quantization.params);
    if (q_params->scale != nullptr && q_params->zero_point != nullptr) {
      q_count = static_cast<size_t>(q_params->scale->size);
      if (q_count > max_quant_params) {
        q_count = max_quant_params;
      }
      for (size_t i = 0; i < q_count; ++i) {
        if (out_scales != nullptr) {
          out_scales[i] = q_params->scale->data[i];
        }
        if (out_zero_points != nullptr) {
          out_zero_points[i] = q_params->zero_point->data[i];
        }
      }
    }
    q_dim = q_params->quantized_dimension;
  }

  if (out_num_quant_params != nullptr) {
    *out_num_quant_params = q_count;
  }
  if (out_quantized_dimension != nullptr) {
    *out_quantized_dimension = q_dim;
  }

  return kTFLMOk;
}

TFLM_Status TFLM_InterpreterGetEvalTensorDetails(
    TFLM_Interpreter* interp, size_t tensor_index, size_t subgraph_index,
    int* out_type, int* out_dims, size_t* out_num_dims, size_t max_dims,
    char* err_buf, size_t err_buf_len) {
  if (interp == nullptr || interp->interpreter == nullptr) {
    SetError(err_buf, err_buf_len, "Null interpreter instance.");
    return kTFLMError;
  }

  if (!interp->interpreter->preserve_all_tensors()) {
    SetError(
        err_buf, err_buf_len,
        "TFLM only supports GetTensor() when using a python interpreter "
        "with the InterpreterConfig.kPreserveAllTensors interpreter_config");
    return kTFLMError;
  }

  const TfLiteEvalTensor* eval_tensor =
      interp->interpreter->GetTensor(tensor_index, subgraph_index);
  if (eval_tensor == nullptr) {
    SetError(err_buf, err_buf_len, "Tensor not found.");
    return kTFLMError;
  }

  if (out_type != nullptr) {
    *out_type = static_cast<int>(eval_tensor->type);
  }

  if (out_dims != nullptr && out_num_dims != nullptr &&
      eval_tensor->dims != nullptr) {
    size_t count = static_cast<size_t>(eval_tensor->dims->size);
    if (count > max_dims) {
      count = max_dims;
    }
    *out_num_dims = count;
    for (size_t i = 0; i < count; ++i) {
      out_dims[i] = eval_tensor->dims->data[i];
    }
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
