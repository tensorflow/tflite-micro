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

#include "tensorflow/lite/micro/tools/python_interpreter/src/interpreter_wrapper.h"

#include <pybind11/embed.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>

#include "numpy.h"
#include "python_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

namespace py = pybind11;

namespace tflite {
namespace interpreter_wrapper {

void* enable_numpy_support() {
  // import_array() is actually a macro that returns NULL (in Python3), hence
  // this wrapper function with a return type of void*.
  import_array();
  return nullptr;
}

InterpreterWrapper::~InterpreterWrapper() {
  delete error_reporter_;
  delete memory_arena_;
  Py_DECREF(model_);
  Py_Finalize();
}

InterpreterWrapper::InterpreterWrapper(PyObject* model_data, int arena_size) {
  // Using model_data as a raw pointer beyond the scope of this constructor, so
  // we need to increment the reference count so that Python doesn't destroy it.
  Py_INCREF(model_data);

  char* buf = nullptr;
  Py_ssize_t length;
  if (python_utils::ConvertFromPyString(model_data, &buf, &length) == -1 ||
      buf == nullptr) {
    throw std::runtime_error(
        "TFLM cannot convert model data from Python object to char *");
  }

  const tflite::Model* model = tflite::GetModel(buf);

  tflite::MicroErrorReporter* micro_error_reporter =
      new tflite::MicroErrorReporter();
  ErrorReporter* error_reporter = micro_error_reporter;

  uint8_t* tensor_arena = new uint8_t[arena_size];
  if (tensor_arena == nullptr) {
    throw std::runtime_error(
        std::string("TFLM cannot malloc memory arena of ") +
        std::to_string(arena_size) + " bytes");
  }

  // Safe to declare here as AllocateTensors() is called here
  tflite::AllOpsResolver all_ops_resolver;

  std::unique_ptr<tflite::MicroInterpreter> interpreter(
      new tflite::MicroInterpreter(model, all_ops_resolver, tensor_arena,
                                   arena_size, error_reporter));

  model_ = model_data;
  error_reporter_ = error_reporter;
  interpreter_ = std::move(interpreter);
  memory_arena_ = tensor_arena;

  TfLiteStatus status = interpreter_->AllocateTensors();
  if (status != kTfLiteOk) {
    throw std::runtime_error("TFLM failed to allocate tensors");
  }

  Py_Initialize();
  enable_numpy_support();
}

void InterpreterWrapper::Invoke() {
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    MicroPrintf("Invoke() failed");
  }
}

void InterpreterWrapper::SetInputTensor(PyObject* data, int index) {
  std::unique_ptr<PyObject, tflite::python_utils::PyDecrefDeleter> array_safe(
      PyArray_FromAny(data, nullptr, 0, 0, NPY_ARRAY_CARRAY, nullptr));
  if (!array_safe) {
    MicroPrintf("Array not safe");
    return;
  }

  // TODO: error checking
  PyArrayObject* array = reinterpret_cast<PyArrayObject*>(array_safe.get());

  memcpy(interpreter_->input(index)->data.data, PyArray_DATA(array),
         PyArray_NBYTES(array));
}

// TODO: Why not use py::array? - it's a bit difficult to deal with the
// templates not knowing types before hand
PyObject* InterpreterWrapper::GetOutputTensor() {
  // Sanity check accessor
  TfLiteTensor* tensor = interpreter_->output(0);
  int type_num = python_utils::TfLiteTypeToPyArrayType(
      tensor->type);  // caused segfault otherwise

  // PyObject* check_result = CheckGetTensorArgs(interpreter_.get(), i, &tensor,
  //                                             &type_num, subgraph_index);
  // if (check_result == nullptr) return check_result;
  // Py_XDECREF(check_result);

  std::vector<npy_intp> dims(tensor->dims->data,
                             tensor->dims->data + tensor->dims->size);
  if (tensor->type != kTfLiteString && tensor->type != kTfLiteResource &&
      tensor->type != kTfLiteVariant) {
    // Make a buffer copy but we must tell Numpy It owns that data or else
    // it will leak.
    void* data = malloc(tensor->bytes);
    if (!data) {
      // PyErr_SetString(PyExc_ValueError, "Malloc to copy tensor failed.");
      MicroPrintf("malloc fails\n");
      return nullptr;
    }
    memcpy(data, tensor->data.raw, tensor->bytes);
    PyObject* np_array;
    if (tensor->sparsity == nullptr) {
      np_array =
          PyArray_SimpleNewFromData(dims.size(), dims.data(), type_num, data);
    } else {
      std::vector<npy_intp> sparse_buffer_dims(1);
      size_t size_of_type = TfLiteTypeGetSize(tensor->type);
      // if (GetSizeOfType(nullptr, tensor->type, &size_of_type) != kTfLiteOk) {
      //   PyErr_SetString(PyExc_ValueError, "Unknown tensor type.");
      //   free(data);
      //   return nullptr;
      // }
      sparse_buffer_dims[0] = tensor->bytes / size_of_type;
      np_array = PyArray_SimpleNewFromData(
          sparse_buffer_dims.size(), sparse_buffer_dims.data(), type_num, data);
    }
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(np_array),
                        NPY_ARRAY_OWNDATA);
    return PyArray_Return(reinterpret_cast<PyArrayObject*>(np_array));
  } else {
    MicroPrintf("unsupported\n");
    // // Create a C-order array so the data is contiguous in memory.
    // const int32_t kCOrder = 0;
    // PyObject* py_object =
    //     PyArray_EMPTY(dims.size(), dims.data(), NPY_OBJECT, kCOrder);

    // if (py_object == nullptr) {
    //   PyErr_SetString(PyExc_MemoryError, "Failed to allocate PyArray.");
    //   return nullptr;
    // }

    // PyArrayObject* py_array = reinterpret_cast<PyArrayObject*>(py_object);
    // PyObject** data = reinterpret_cast<PyObject**>(PyArray_DATA(py_array));
    // auto num_strings = GetStringCount(tensor);
    // for (int j = 0; j < num_strings; ++j) {
    //   auto ref = GetString(tensor, j);

    //   PyObject* bytes = PyBytes_FromStringAndSize(ref.str, ref.len);
    //   if (bytes == nullptr) {
    //     Py_DECREF(py_object);
    //     PyErr_Format(PyExc_ValueError,
    //                  "Could not create PyBytes from string %d of input %d.",
    //                  j, i);
    //     return nullptr;
    //   }
    //   // PyArray_EMPTY produces an array full of Py_None, which we must
    //   decref. Py_DECREF(data[j]); data[j] = bytes;
    // }
    // return py_object;
    return nullptr;
  }
}

}  // namespace interpreter_wrapper
}  // namespace tflite
