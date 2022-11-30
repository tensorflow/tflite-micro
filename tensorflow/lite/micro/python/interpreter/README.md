# TFLM Python Interpreter

The TFLM interpreter can be invoked from Python by using the Python interpreter
wrapper in this directory.

## Usage

There are two ways to import the Python wrapper, either by using Bazel/Blaze, or
in near future by installing a PyPi package.

### Bazel

#### Build

The only package that needs to be included in the `BUILD` file is
`//tensorflow/lite/micro/python/interpreter/src:tflm_runtime`. It contains all
the correct dependencies to build the Python interpreter.

### PyPi

Work in progress.

### Examples

Depending on the workflow, the package import path may be slightly different.

A simple end-to-end example is the test
`tensorflow/lite/micro/python/interpreter/tests/interpreter_test.py:testCompareWithTFLite()`.
It shows how to compare inference results between TFLite and TFLM.

A basic usage of the TFLM Python interpreter looks like the following. The input
to the Python interpreter should be a converted TFLite flatbuffer in either
bytearray format or file format.

```
# For the Bazel workflow
from tflite_micro.tensorflow.lite.micro.python.interpreter.src import tflm_runtime


# If model is a bytearray
tflm_interpreter = tflm_runtime.Interpreter.from_bytes(model_data)
# If model is a file
tflm_interpreter = tflm_runtime.Interpreter.from_file(model_filepath)

# Run inference on TFLM using an ndarray `data_x`
tflm_interpreter.set_input(data_x, 0)
tflm_interpreter.invoke()
tflm_output = tflm_interpreter.get_output(0)
```

Input and output tensor details can also be queried using the Python API:

```
print(tflm_interpreter.get_input_details[0])
print(tflm_interpreter.get_output_details[0])
```

## Technical Details

The Python interpreter uses [pybind11](https://github.com/pybind/pybind11) to
expose an evolving set of C++ APIs. The Bazel build leverages the
[pybind11_bazel extension](https://github.com/pybind/pybind11_bazel).

The most updated Python APIs can be found in
`tensorflow/lite/micro/python/interpreter/src/tflm_runtime.py`.

## Custom Ops

The Python interpreter works with models with
[custom ops](https://www.tensorflow.org/lite/guide/ops_custom) but special steps
need to be taken to make sure that it can retrieve the right implementation.
This is currently compatible with the Bazel workflow only.

1. Implement the custom op in C++

Assuming that the custom is already implemented according to the linked guide,

```
// custom_op.cc
TfLiteRegistration *Register_YOUR_CUSTOM_OP() {
    // Do custom op stuff
}

// custom_op.h
TfLiteRegistration *Register_YOUR_CUSTOM_OP();
```

2. Implement a custom op Registerer

A Registerer of the following signature is required to wrap the custom op and
add it to TFLM's ops resolver. For example,

```
#include "custom_op.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

namespace tflite {

extern "C" bool SomeCustomRegisterer(tflite::AllOpsResolver* resolver) {
    TfLiteStatus status = resolver->AddCustom("CustomOp", tflite::Register_YOUR_CUSTOM_OP());
    if (status != kTfLiteOk) {
        return false;
    }
    return true;
}
```

3. Include the implementation of custom op and registerer in the caller's build

For the Bazel workflow, it's recommended to create a package that includes the
custom op's and the registerer's implementation, because it needs to be included
in the target that calls the Python interpreter with custom ops.

4. Pass the registerer into the Python interpreter during instantiation

For example,

```
interpreter = tflm_runtime.Interpreter.from_file(
    model_path=model_path,
    custom_op_registerers=['SomeCustomRegisterer'])
```

The interpreter will then perform a dynamic lookup for the symbol called
`SomeCustomRegisterer()` and call it. This ensures that the custom op is
properly included in TFLM's op resolver. This approach is very similar to
TFLite's custom op support.

## Print Allocations

The Python interpreter can also be used to print memory arena allocations. This
is very helpful to figure out actual memory arena usage.

For example,

```
tflm_interpreter.print_allocations()
```

will print

```
[RecordingMicroAllocator] Arena allocation total 10016 bytes
[RecordingMicroAllocator] Arena allocation head 7744 bytes
[RecordingMicroAllocator] Arena allocation tail 2272 bytes
[RecordingMicroAllocator] 'TfLiteEvalTensor data' used 312 bytes with alignment overhead (requested 312 bytes for 13 allocations)
[RecordingMicroAllocator] 'Persistent TfLiteTensor data' used 224 bytes with alignment overhead (requested 224 bytes for 2 tensors)
[RecordingMicroAllocator] 'Persistent TfLiteTensor quantization data' used 64 bytes with alignment overhead (requested 64 bytes for 4 allocations)
[RecordingMicroAllocator] 'Persistent buffer data' used 640 bytes with alignment overhead (requested 608 bytes for 10 allocations)
[RecordingMicroAllocator] 'NodeAndRegistration struct' used 440 bytes with alignment overhead (requested 440 bytes for 5 NodeAndRegistration structs)
```

10016 bytes is the actual memory arena size.

During instantiation via the class methods `tflm_runtime.Interpreter.from_file`
or `tflm_runtime.Interpreter.from_bytes`, if `arena_size` is not explicitly
specified, the interpreter will default to a heuristic which is 10x the model
size. This can be adjusted manually if desired.
