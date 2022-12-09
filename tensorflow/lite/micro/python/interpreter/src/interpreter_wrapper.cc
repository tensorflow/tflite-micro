/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/python/interpreter/src/interpreter_wrapper.h"

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// See https://numpy.org/doc/1.16/reference/c-api.array.html#importing-the-api
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL tflite_micro_python_interpreter_array_api
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/python/interpreter/src/numpy_utils.h"
#include "tensorflow/lite/micro/python/interpreter/src/pybind11_lib.h"
#include "tensorflow/lite/micro/python/interpreter/src/python_utils.h"
#include "tensorflow/lite/micro/python/interpreter/src/shared_library.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace tflite {
namespace {
// This function looks up the registerer symbol based on the string name
// `registerer_name`. A registerer in this case is a function that calls the
// `AddCustom` API of `AllOpsResolver` for custom ops that need to be registered
// with the interpreter.
bool AddCustomOpRegistererByName(const char* registerer_name,
                                 tflite::AllOpsResolver* resolver) {
  // Registerer functions take a pointer to a AllOpsResolver as an input
  // parameter and return TfLiteStatus.
  typedef bool (*RegistererFunctionType)(tflite::AllOpsResolver*);

  // Look for the Registerer function by name.
  RegistererFunctionType registerer = reinterpret_cast<RegistererFunctionType>(
      SharedLibrary::GetSymbol(registerer_name));

  // Fail in an informative way if the function was not found.
  if (registerer == nullptr) {
    MicroPrintf("Looking up symbol '%s' failed with error '%s'.",
                registerer_name, SharedLibrary::GetError());
    return false;
  }

  // Call the registerer with the resolver.
  if (!registerer(resolver)) {
    MicroPrintf(
        "%s failed to register op. Check that total number of "
        "ops doesn't exceed the maximum allowed by AllOpsResolver.",
        registerer_name);
    return false;
  }

  return true;
}

PyObject* PyArrayFromFloatVector(const float* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(float));
  memcpy(pydata, data, size * sizeof(float));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_FLOAT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

PyObject* PyArrayFromIntVector(const int* data, npy_intp size) {
  void* pydata = malloc(size * sizeof(int));
  memcpy(pydata, data, size * sizeof(int));
  PyObject* obj = PyArray_SimpleNewFromData(1, &size, NPY_INT32, pydata);
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(obj), NPY_ARRAY_OWNDATA);
  return obj;
}

// Check if the tensor is valid for TFLM
bool CheckTensor(const TfLiteTensor* tensor) {
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_IndexError,
                    "Tensor is out of bound, please check tensor index.");
    return false;
  }

  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    PyErr_SetString(PyExc_ValueError,
                    "TFLM doesn't support strings, resource variables, or "
                    "variants as outputs.");
    return false;
  }

  if (tensor->sparsity != nullptr) {
    PyErr_SetString(PyExc_ValueError, "TFLM doesn't support sparse tensors");
    return false;
  }

  int py_type_num = TfLiteTypeToPyArrayType(tensor->type);
  if (py_type_num == NPY_NOTYPE) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
    return false;
  }

  if (tensor->bytes == 0 && tensor->data.data != nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid tensor size of 0.");
    return false;
  }

  if (tensor->bytes > 0 && tensor->data.data == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Null tensor pointer.");
    return false;
  }
  return true;
}

PyObject* GetTensorSize(const TfLiteTensor* tensor) {
  PyObject* np_array =
      PyArrayFromIntVector(tensor->dims->data, tensor->dims->size);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* GetTensorType(const TfLiteTensor* tensor) {
  int code = TfLiteTypeToPyArrayType(tensor->type);
  return PyArray_TypeObjectFromType(code);
}

// Create a python dictionary object that contains the general (can be
// channel-wise quantized) affiene quantization information about the tensor.
PyObject* GetTensorQuantizationParameters(const TfLiteTensor* tensor) {
  const TfLiteQuantization quantization = tensor->quantization;
  float* scales_data = nullptr;
  int32_t* zero_points_data = nullptr;
  int32_t scales_size = 0;
  int32_t zero_points_size = 0;
  int32_t quantized_dimension = 0;
  if (quantization.type == kTfLiteAffineQuantization) {
    const TfLiteAffineQuantization* q_params =
        reinterpret_cast<const TfLiteAffineQuantization*>(quantization.params);
    if (q_params->scale) {
      scales_data = q_params->scale->data;
      scales_size = q_params->scale->size;
    }
    if (q_params->zero_point) {
      zero_points_data = q_params->zero_point->data;
      zero_points_size = q_params->zero_point->size;
    }
    quantized_dimension = q_params->quantized_dimension;
  }
  PyObject* scales_array = PyArrayFromFloatVector(scales_data, scales_size);
  PyObject* zero_points_array =
      PyArrayFromIntVector(zero_points_data, zero_points_size);

  PyObject* result = PyDict_New();
  PyDict_SetItemString(result, "scales", scales_array);
  PyDict_SetItemString(result, "zero_points", zero_points_array);
  PyDict_SetItemString(result, "quantized_dimension",
                       PyLong_FromLong(quantized_dimension));
  return result;
}

PyObject* GetTensorDetails(const TfLiteTensor* tensor) {
  if (!CheckTensor(tensor)) {
    return nullptr;
  }

  PyObject* tensor_type = GetTensorType(tensor);
  PyObject* tensor_size = GetTensorSize(tensor);
  PyObject* tensor_quantization_parameters =
      GetTensorQuantizationParameters(tensor);

  PyObject* result = PyDict_New();
  PyDict_SetItemString(result, "dtype", tensor_type);
  PyDict_SetItemString(result, "shape", tensor_size);
  PyDict_SetItemString(result, "quantization_parameters",
                       tensor_quantization_parameters);

  return result;
}

}  // namespace

InterpreterWrapper::~InterpreterWrapper() {
  // We don't use a unique_ptr for the interpreter because we need to call its
  // destructor before we call Py_DECREF(model_). This ensures that the model
  // is still in scope when MicroGraph:FreeSubgraphs() is called. Otherwise,
  // a segmentation fault could occur.
  if (interpreter_ != nullptr) {
    delete interpreter_;
  }

  // Undo any references incremented
  Py_DECREF(model_);
}

InterpreterWrapper::InterpreterWrapper(
    PyObject* model_data, const std::vector<std::string>& registerers_by_name,
    size_t arena_size, int num_resource_variables) {
  interpreter_ = nullptr;

  // `model_data` is used as a raw pointer beyond the scope of this
  // constructor, so we need to increment the reference count so that Python
  // doesn't destroy it during the lifetime of this interpreter.
  Py_INCREF(model_data);

  // Get the input array contained in `model_data` as a byte array
  char* buf = nullptr;
  Py_ssize_t length;
  if (ConvertFromPyString(model_data, &buf, &length) == -1 || buf == nullptr) {
    ThrowValueError(
        "TFLM cannot convert model data from Python object to char *");
  }

  const Model* model = GetModel(buf);
  model_ = model_data;
  memory_arena_ = std::unique_ptr<uint8_t[]>(new uint8_t[arena_size]);
  allocator_ = RecordingMicroAllocator::Create(memory_arena_.get(), arena_size);
  MicroResourceVariables* resource_variables_ = nullptr;
  if (num_resource_variables > 0)
    resource_variables_ =
        MicroResourceVariables::Create(allocator_, num_resource_variables);

  for (const auto& registerer : registerers_by_name) {
    if (!AddCustomOpRegistererByName(registerer.c_str(), &all_ops_resolver_)) {
      char strbuf[128];
      snprintf(strbuf, sizeof(strbuf),
               "TFLM could not register custom op via %s", registerer.c_str());
      ThrowRuntimeError(strbuf);
    }
  }

  interpreter_ = new MicroInterpreter(model, all_ops_resolver_, allocator_,
                                      resource_variables_);

  TfLiteStatus status = interpreter_->AllocateTensors();
  if (status != kTfLiteOk) {
    ThrowRuntimeError("TFLM failed to allocate tensors");
  }

  // This must be called before using any PyArray_* APIs. It essentially sets
  // up the lookup table that maps PyArray_* macros to the correct APIs.
  ImportNumpy();
}

void InterpreterWrapper::PrintAllocations() { allocator_->PrintAllocations(); }

int InterpreterWrapper::Invoke() { return interpreter_->Invoke(); }

int InterpreterWrapper::Reset() { return interpreter_->Reset(); }

// 1. Check that tensor and input array are safe to access
// 2. Verify that input array metadata matches tensor metadata
// 3. Copy input buffer into target input tensor
void InterpreterWrapper::SetInputTensor(PyObject* data, size_t index) {
  std::unique_ptr<PyObject, PyDecrefDeleter> array_safe(PyArray_FromAny(
      /*op=*/data,
      /*dtype=*/nullptr,
      /*min_depth=*/0,
      /*max_depth=*/0,
      /*requirements=*/NPY_ARRAY_CARRAY,
      /*context=*/nullptr));
  if (!array_safe) {
    ThrowValueError("TFLM cannot convert input to PyArray");
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  TfLiteTensor* tensor = interpreter_->input(index);
  if (!CheckTensor(tensor)) {
    throw pybind11::error_already_set();
  }

  char err_strbuf[128];  // buffer for error message since we can not use
                         // absl
  if (TfLiteTypeFromPyArray(array) != tensor->type) {
    snprintf(err_strbuf, sizeof(err_strbuf),
             "Cannot set tensor: Got value of type %s but expected type %s "
             "for input %zu ",
             TfLiteTypeGetName(TfLiteTypeFromPyArray(array)),
             TfLiteTypeGetName(tensor->type), index);

    ThrowValueError(err_strbuf);
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    snprintf(err_strbuf, sizeof(err_strbuf),
             "Cannot set tensor: Dimension mismatch. Got %d but expected "
             "%d for input %zu.",
             PyArray_NDIM(array), tensor->dims->size, index);
    ThrowValueError(err_strbuf);
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      snprintf(err_strbuf, sizeof(err_strbuf),
               "Cannot set tensor: Dimension mismatch. Got %ld but "
               "expected %d for dimension %d of input %zu.",
               PyArray_SHAPE(array)[j], tensor->dims->data[j], j, index);
      ThrowValueError(err_strbuf);
    }
  }

  if (tensor->data.data == nullptr && tensor->bytes) {
    ThrowValueError("Cannot set tensor: Tensor is non-empty but has nullptr.");
  }

  size_t size = PyArray_NBYTES(array);
  if (size != tensor->bytes) {
    snprintf(err_strbuf, sizeof(err_strbuf),
             "numpy array had %zu bytes but expected %zu bytes.", size,
             tensor->bytes);
    ThrowValueError(err_strbuf);
  }

  memcpy(tensor->data.data, PyArray_DATA(array), size);
}

// 1. Check that output tensor is supported and safe to access
// 2. Allocate a buffer and copy output tensor data into it
// 3. Set PyArray metadata and transfer ownership to caller
PyObject* InterpreterWrapper::GetOutputTensor(size_t index) const {
  const TfLiteTensor* tensor = interpreter_->output(index);
  if (!CheckTensor(tensor)) {
    return nullptr;
  }
  // Allocate a new buffer with output data to be returned to Python. New memory
  // is allocated here to prevent hard to debug issues in Python, like data
  // potentially changing under the hood, which imposes an implicit requirement
  // that the user needs to be aware of.
  void* data = malloc(tensor->bytes);
  memcpy(data, tensor->data.data, tensor->bytes);

  PyObject* np_array;
  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  int py_type_num = TfLiteTypeToPyArrayType(tensor->type);
  np_array =
      PyArray_SimpleNewFromData(dims.size(), dims.data(), py_type_num, data);

  // Transfer ownership to Python so that there's Python will take care of
  // releasing this buffer
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                      NPY_ARRAY_OWNDATA);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

PyObject* InterpreterWrapper::GetInputTensorDetails(size_t index) const {
  return GetTensorDetails(interpreter_->input(index));
}

PyObject* InterpreterWrapper::GetOutputTensorDetails(size_t index) const {
  return GetTensorDetails(interpreter_->output(index));
}

}  // namespace tflite
