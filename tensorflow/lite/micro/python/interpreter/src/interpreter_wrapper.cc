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
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/python/interpreter/src/numpy_utils.h"
#include "tensorflow/lite/micro/python/interpreter/src/python_utils.h"
#include "tensorflow/lite/micro/python/interpreter/src/shared_library.h"

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
    size_t arena_size) {
  interpreter_ = nullptr;

  // `model_data` is used as a raw pointer beyond the scope of this
  // constructor, so we need to increment the reference count so that Python
  // doesn't destroy it during the lifetime of this interpreter.
  Py_INCREF(model_data);

  // Get the input array contained in `model_data` as a byte array
  char* buf = nullptr;
  Py_ssize_t length;
  if (ConvertFromPyString(model_data, &buf, &length) == -1 || buf == nullptr) {
    PyErr_SetString(
        PyExc_ValueError,
        "TFLM cannot convert model data from Python object to char *");
    return;
  }

  const Model* model = GetModel(buf);
  model_ = model_data;
  error_reporter_ = std::unique_ptr<ErrorReporter>(new MicroErrorReporter());
  memory_arena_ = std::unique_ptr<uint8_t[]>(new uint8_t[arena_size]);

  for (const auto& registerer : registerers_by_name) {
    if (!AddCustomOpRegistererByName(registerer.c_str(), &all_ops_resolver_)) {
      char strbuf[128];
      snprintf(strbuf, sizeof(strbuf),
               "TFLM could not register custom op via %s", registerer.c_str());
      PyErr_SetString(PyExc_RuntimeError, strbuf);
      return;
    }
  }

  interpreter_ =
      new MicroInterpreter(model, all_ops_resolver_, memory_arena_.get(),
                           arena_size, error_reporter_.get());

  TfLiteStatus status = interpreter_->AllocateTensors();
  if (status != kTfLiteOk) {
    PyErr_SetString(PyExc_RuntimeError, "TFLM failed to allocate tensors");
    return;
  }

  // This must be called before using any PyArray_* APIs. It essentially sets
  // up the lookup table that maps PyArray_* macros to the correct APIs.
  ImportNumpy();
}

int InterpreterWrapper::Invoke() { return interpreter_->Invoke(); }

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
    PyErr_SetString(PyExc_ValueError, "TFLM cannot convert input to PyArray");
    return;
  }

  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  TfLiteTensor* tensor = interpreter_->input(index);
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_IndexError,
                    "Tensor is out of bound, please check tensor index.");
    return;
  }

  if (tensor->type == kTfLiteString) {
    PyErr_SetString(PyExc_ValueError, "TFLM does not support string input.");
    return;
  }

  if (TfLiteTypeFromPyArray(array) != tensor->type) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Got value of type %s but expected type %s "
                 "for input %zu, name: %s ",
                 TfLiteTypeGetName(TfLiteTypeFromPyArray(array)),
                 TfLiteTypeGetName(tensor->type), index, tensor->name);
    return;
  }

  if (PyArray_NDIM(array) != tensor->dims->size) {
    PyErr_Format(PyExc_ValueError,
                 "Cannot set tensor: Dimension mismatch. Got %d but expected "
                 "%d for input %zu.",
                 PyArray_NDIM(array), tensor->dims->size, index);
    return;
  }

  for (int j = 0; j < PyArray_NDIM(array); j++) {
    if (tensor->dims->data[j] != PyArray_SHAPE(array)[j]) {
      PyErr_Format(PyExc_ValueError,
                   "Cannot set tensor: Dimension mismatch. Got %ld but "
                   "expected %d for dimension %d of input %zu.",
                   PyArray_SHAPE(array)[j], tensor->dims->data[j], j, index);
      return;
    }
  }

  if (tensor->data.data == nullptr && tensor->bytes) {
    PyErr_Format(PyExc_RuntimeError,
                 "Cannot set tensor: Tensor is non-empty but has nullptr.");
    return;
  }

  size_t size = PyArray_NBYTES(array);
  if (size != tensor->bytes) {
    PyErr_Format(PyExc_ValueError,
                 "numpy array had %zu bytes but expected %zu bytes.", size,
                 tensor->bytes);
    return;
  }

  memcpy(tensor->data.data, PyArray_DATA(array), size);
}

// 1. Check that output tensor is supported and safe to access
// 2. Allocate a buffer and copy output tensor data into it
// 3. Set PyArray metadata and transfer ownership to caller
PyObject* InterpreterWrapper::GetOutputTensor(size_t index) {
  TfLiteTensor* tensor = interpreter_->output(index);
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_IndexError, "Tensor is out of bound.");
    return nullptr;
  }

  if (tensor->type == kTfLiteString || tensor->type == kTfLiteResource ||
      tensor->type == kTfLiteVariant) {
    PyErr_SetString(PyExc_ValueError,
                    "TFLM doesn't support strings, resource variables, or "
                    "variants as outputs.");
    return nullptr;
  }

  if (tensor->sparsity != nullptr) {
    PyErr_SetString(PyExc_ValueError, "TFLM doesn't support sparse tensors");
    return nullptr;
  }

  int py_type_num = TfLiteTypeToPyArrayType(tensor->type);
  if (py_type_num == NPY_NOTYPE) {
    PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
    return nullptr;
  }

  if (tensor->bytes == 0 && tensor->data.data != nullptr) {
    PyErr_SetString(PyExc_ValueError, "Invalid tensor size of 0.");
    return nullptr;
  }

  if (tensor->bytes > 0 && tensor->data.data == nullptr) {
    PyErr_SetString(PyExc_ValueError, "Null tensor pointer.");
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
  np_array =
      PyArray_SimpleNewFromData(dims.size(), dims.data(), py_type_num, data);

  // Transfer ownership to Python so that there's Python will take care of
  // releasing this buffer
  PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                      NPY_ARRAY_OWNDATA);

  return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
}

}  // namespace tflite
