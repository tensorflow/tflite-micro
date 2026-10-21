# The `tflite_micro` Python Package

This directory contains the `tflite_micro` Python package. The following is
mainly documentation for its developers.

The `tflite_micro` package contains a complete TFLM interpreter built as a
CPython extension module. The build of simple Python packages may be driven by
standard Python package builders such as `build`, `setuptools`, and `flit`;
however, as TFLM is first and foremost a large C/C++ project, `tflite_micro`'s
build is instead driven by its C/C++ build system Bazel.

## Building and installing locally

### Building

The Bazel target `//python/tflite_micro:whl.dist` builds a `tflite_micro`
Python *.whl* under the output directory `bazel-bin/python/tflite_micro/whl_dist`. For example:
```
% bazel build //python/tflite_micro:whl.dist
....
Target //python/tflite_micro:whl.dist up-to-date:
  bazel-bin/python/tflite_micro/whl_dist

% tree bazel-bin/python/tflite_micro/whl_dist
bazel-bin/python/tflite_micro/whl_dist
└── tflite_micro-0.dev20230920161638-py3-none-any.whl
```

### Installing

Install the resulting *.whl* via pip. For example, in a Python virtual
environment:
```
% python3 -m venv ~/tmp/venv
% source ~/tmp/venv/bin/activate
(venv) $ pip install bazel-bin/python/tflite_micro/whl_dist/tflite_micro-0.dev20230920161638-py3-none-any.whl
Processing ./bazel-bin/python/tflite_micro/whl_dist/tflite_micro-0.dev20230920161638-py3-none-any.whl
....
Installing collected packages: [....]
```

The package should now be importable and usable. For example:
```
(venv) $ python
Python 3.10.12 (main, Jun 11 2023, 05:26:28) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tflite_micro
>>> tflite_micro.postinstall_check.passed()
True
>>>  i = tflite_micro.runtime.Interpreter.from_file("foo.tflite")
>>> # etc.
```

## Building and uploading to PyPI

The *.whl* generated above is unsuitable for distribution to the wider world
via PyPI. The extension module is inevitably compiled against a particular
Python implementation and platform C library. The resulting package is only
binary-compatible with a system running the same Python implementation and a
compatible (typically the same or newer) C library.

The solution is to distribute multiple *.whl*s, one built for each Python
implementation and platform combination. TFLM accomplishes this by running
Bazel builds from within multiple, uniquely configured Docker containers. The
images used are based on standards-conforming images published by the Python
Package Authority (PyPA) for exactly such use.

Python *.whl*s contain metadata used by installers such as `pip` to determine
which distributions (*.whl*s) are compatible with the target platform. See the PyPA
specification for [platform compatibility
tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).

### Building

In an environment with a working Docker installation, run the script
`python/tflite_micro/pypi_build.sh`. The script builds an
appropriate Docker container and invokes a Bazel build and test within it, producing
a universal Python 3 wheel (`py3-none-manylinux_...`):
```
% python/tflite_micro/pypi_build.sh
...
Output:
bazel-pypi-out/tflite_micro-0.dev20261020031310-py3-none-manylinux_2_28_x86_64.whl
```

By default, *.whl*s are generated under the output directory `bazel-pypi-out/`.

### Uploading to PyPI

Upload the generated *.whl* to PyPI with `twine`. A PyPI authentication token must be assigned to
`TWINE_PASSWORD` in the environment. For example:
```
% export TWINE_PASSWORD=pypi-AgENdGV[....]
% twine upload bazel-pypi-out/tflite_micro-*.whl
```

See the script's online help (`--help`) for more.

## Using `tflite_micro` from within the TFLM source tree

:construction:
*The remainder of this document is under construction and may contain some
obsolete information.*
:construction:

The only package that needs to be included in the `BUILD` file is
`//python/tflite_micro:runtime`. It contains all
the correct dependencies to build the Python interpreter.

### Examples

Depending on the workflow, the package import path may be slightly different.

A simple end-to-end example is the test
`python/tflite_micro/runtime_test.py:testCompareWithTFLite()`.
It shows how to compare inference results between TFLite and TFLM.

A basic usage of the TFLM Python interpreter looks like the following. The input
to the Python interpreter should be a converted TFLite flatbuffer in either
bytearray format or file format.

```
# For the Bazel workflow
from tflite_micro.python.tflite_micro import runtime


# If model is a bytearray
tflm_interpreter = runtime.Interpreter.from_bytes(model_data)
# If model is a file
tflm_interpreter = runtime.Interpreter.from_file(model_filepath)

# Run inference on TFLM using an ndarray `data_x`
tflm_interpreter.set_input(data_x, 0)
tflm_interpreter.invoke()
tflm_output = tflm_interpreter.get_output(0)
```

Input and output tensor details can also be queried using the Python API:

```
print(tflm_interpreter.get_input_details(0))
print(tflm_interpreter.get_output_details(0))
```

### Tutorials

For a complete end-to-end example using the Python interpreter for model compression:
* [MNIST Compression Tutorial](../../tensorflow/lite/micro/compression/tutorial/mnist_compression_tutorial.ipynb)

### Technical Details

The Python interpreter is built as a pure C-ABI shared library (`libtflm.so`) that
interfaces with Python via Python's standard library `ctypes`. This avoids any
dependency on CPython internal headers or binding libraries, making the compiled
binary forward-compatible across all Python 3 versions without rebuilding.

The most updated Python APIs can be found in
`python/tflite_micro/runtime.py`.

### Custom Ops

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

extern "C" bool SomeCustomRegisterer(tflite::PythonOpsResolver* resolver) {
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
interpreter = runtime.Interpreter.from_file(
    model_path=model_path,
    custom_op_registerers=['SomeCustomRegisterer'])
```

The interpreter will then perform a dynamic lookup for the symbol called
`SomeCustomRegisterer()` and call it. This ensures that the custom op is
properly included in TFLM's op resolver. This approach is very similar to
TFLite's custom op support.

### Print Allocations

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

During instantiation via the class methods `runtime.Interpreter.from_file`
or `runtime.Interpreter.from_bytes`, if `arena_size` is not explicitly
specified, the interpreter will default to a heuristic which is 10x the model
size. This can be adjusted manually if desired.
