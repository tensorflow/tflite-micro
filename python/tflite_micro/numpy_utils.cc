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

#include "python/tflite_micro/numpy_utils.h"

// Disallow Numpy 1.7 deprecated symbols.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// Since we are calling `import_array()` here, define PY_ARRAY_UNIQUE_SYMBOL
// here and NO_IMPORT_ARRAY everywhere else arrayobject.h is included
// See https://numpy.org/doc/1.16/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL tflite_micro_python_interpreter_array_api
#include <numpy/arrayobject.h>

#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {

void* ImportNumpy() {
  // import_array() is actually a macro that returns NULL (in Python3), hence
  // this wrapper function with a return type of void*.
  import_array();
  return nullptr;
}

int TfLiteTypeToPyArrayType(TfLiteType tf_lite_type) {
  switch (tf_lite_type) {
    case kTfLiteFloat32:
      return NPY_FLOAT32;
    case kTfLiteFloat16:
      return NPY_FLOAT16;
    case kTfLiteBFloat16:
      // TODO(b/329491949): Supports other ml_dtypes user-defined types.
      return NPY_USERDEF;
    case kTfLiteFloat64:
      return NPY_FLOAT64;
    case kTfLiteInt32:
      return NPY_INT32;
    case kTfLiteUInt32:
      return NPY_UINT32;
    case kTfLiteUInt16:
      return NPY_UINT16;
    case kTfLiteInt16:
      return NPY_INT16;
    case kTfLiteUInt8:
      return NPY_UINT8;
    case kTfLiteInt4:
      // TODO(b/246806634): NPY_INT4 currently doesn't exist
      return NPY_BYTE;
    case kTfLiteInt8:
      return NPY_INT8;
    case kTfLiteInt64:
      return NPY_INT64;
    case kTfLiteUInt64:
      return NPY_UINT64;
    case kTfLiteString:
      return NPY_STRING;
    case kTfLiteBool:
      return NPY_BOOL;
    case kTfLiteComplex64:
      return NPY_COMPLEX64;
    case kTfLiteComplex128:
      return NPY_COMPLEX128;
    case kTfLiteResource:
    case kTfLiteVariant:
      return NPY_OBJECT;
    case kTfLiteNoType:
      return NPY_NOTYPE;
      // Avoid default so compiler errors created when new types are made.
  }
  return NPY_NOTYPE;
}

TfLiteType TfLiteTypeFromPyType(int py_type) {
  switch (py_type) {
    case NPY_FLOAT32:
      return kTfLiteFloat32;
    case NPY_FLOAT16:
      return kTfLiteFloat16;
    case NPY_FLOAT64:
      return kTfLiteFloat64;
    case NPY_INT32:
      return kTfLiteInt32;
    case NPY_UINT32:
      return kTfLiteUInt32;
    case NPY_INT16:
      return kTfLiteInt16;
    case NPY_UINT8:
      return kTfLiteUInt8;
    case NPY_INT8:
      return kTfLiteInt8;
    case NPY_INT64:
      return kTfLiteInt64;
    case NPY_UINT64:
      return kTfLiteUInt64;
    case NPY_BOOL:
      return kTfLiteBool;
    case NPY_OBJECT:
    case NPY_STRING:
    case NPY_UNICODE:
      return kTfLiteString;
    case NPY_COMPLEX64:
      return kTfLiteComplex64;
    case NPY_COMPLEX128:
      return kTfLiteComplex128;
    case NPY_USERDEF:
      // User-defined types are defined in ml_dtypes. (bfloat16, float8, etc.)
      // Fow now, we only support bfloat16.
      return kTfLiteBFloat16;
      // Avoid default so compiler errors created when new types are made.
  }
  return kTfLiteNoType;
}

TfLiteType TfLiteTypeFromPyArray(const PyArrayObject* array) {
  int pyarray_type = PyArray_TYPE(array);
  return TfLiteTypeFromPyType(pyarray_type);
}

}  // namespace tflite
