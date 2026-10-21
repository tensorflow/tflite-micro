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

#ifndef PYTHON_TFLITE_MICRO_RUNTIME_C_API_H_
#define PYTHON_TFLITE_MICRO_RUNTIME_C_API_H_

/*
 * WARNING: This is a private C interface specifically for the Python ctypes
 * bindings (python/tflite_micro/runtime.py). It is NOT a public C API for
 * general consumption, and its interface is subject to change at any time.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#ifdef TFLM_COMPILE_LIBRARY
#define TFLM_PYTHON_C_API_EXPORT __declspec(dllexport)
#else
#define TFLM_PYTHON_C_API_EXPORT __declspec(dllimport)
#endif  // TFLM_COMPILE_LIBRARY
#else
#define TFLM_PYTHON_C_API_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

typedef struct TFLM_Interpreter TFLM_Interpreter;

// NOLINTNEXTLINE(performance-enum-size)
typedef enum {
  kTFLMOk = 0,
  kTFLMError = 1,
} TFLM_Status;

// NOLINTNEXTLINE(performance-enum-size)
typedef enum {
  kTFLMConfigAllocationRecording = 0,
  kTFLMConfigPreserveAllTensors = 1,
} TFLM_InterpreterConfig;

// Create a new TFLM Interpreter.
// Returns nullptr on failure and populates err_buf if provided.
TFLM_PYTHON_C_API_EXPORT TFLM_Interpreter* TFLM_InterpreterCreate(
    const uint8_t* model_data, size_t model_size,
    const char* const* custom_op_registerers, size_t num_registerers,
    size_t arena_size, int num_resource_variables, int interpreter_config,
    size_t alt_decompression_memory_size, char* err_buf, size_t err_buf_len);

// Destroy the TFLM Interpreter.
TFLM_PYTHON_C_API_EXPORT void TFLM_InterpreterDestroy(TFLM_Interpreter* interp);

// Execution
TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_InterpreterInvoke(
    TFLM_Interpreter* interp, char* err_buf, size_t err_buf_len);

TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_InterpreterReset(
    TFLM_Interpreter* interp, char* err_buf, size_t err_buf_len);

TFLM_PYTHON_C_API_EXPORT void TFLM_InterpreterPrintAllocations(
    TFLM_Interpreter* interp);

// Input & Output Tensor Data
TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_InterpreterSetInputData(
    TFLM_Interpreter* interp, size_t index, const void* data, size_t byte_size,
    int type_code, const int* dims, size_t num_dims, char* err_buf,
    size_t err_buf_len);

// Returns pointer to output tensor buffer inside arena and sets out_byte_size.
TFLM_PYTHON_C_API_EXPORT const void* TFLM_InterpreterGetOutputData(
    const TFLM_Interpreter* interp, size_t index, size_t* out_byte_size,
    char* err_buf, size_t err_buf_len);

// For preserve_all_tensors mode
TFLM_PYTHON_C_API_EXPORT const void* TFLM_InterpreterGetTensorData(
    TFLM_Interpreter* interp, size_t tensor_index, size_t subgraph_index,
    size_t* out_byte_size, char* err_buf, size_t err_buf_len);

// Metadata & Quantization Details
TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_InterpreterGetTensorDetails(
    const TFLM_Interpreter* interp, int is_input, size_t index, int* out_type,
    int* out_dims, size_t* out_num_dims, size_t max_dims, float* out_scales,
    int32_t* out_zero_points, size_t* out_num_quant_params,
    size_t max_quant_params, int* out_quantized_dimension, char* err_buf,
    size_t err_buf_len);

TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_InterpreterGetEvalTensorDetails(
    TFLM_Interpreter* interp, size_t tensor_index, size_t subgraph_index,
    int* out_type, int* out_dims, size_t* out_num_dims, size_t max_dims,
    char* err_buf, size_t err_buf_len);

// Model alignment helper
TFLM_PYTHON_C_API_EXPORT TFLM_Status
TFLM_AlignFlatbuffer(const char* input_path, const char* output_path,
                     char* err_buf, size_t err_buf_len);

// Signal utils helpers
TFLM_PYTHON_C_API_EXPORT float TFLM_FreqToMel(float freq);

TFLM_PYTHON_C_API_EXPORT TFLM_Status TFLM_WideDynamicFuncLut(
    float strength, float offset, int input_bits, int gain_bits,
    int16_t* out_lut, size_t max_lut_size, size_t* out_lut_size);

#ifdef __cplusplus
}
#endif

#endif  // PYTHON_TFLITE_MICRO_RUNTIME_C_API_H_
