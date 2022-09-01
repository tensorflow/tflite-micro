<!--ts-->
   * [Background](#background)
   * [Data Files in Examples](#data-files-in-examples)

<!-- Added by: njeff, at: Wed Nov 17 11:33:14 AM PST 2021 -->

<!--te-->

# Background

TFLM is designed to run on microcontrollers and other platforms without dynamic
memory allocation and without filesystems. This means that data files such as
TFLite models and test inputs must be built into the binary.

Historically, data files have been included as cc arrays generated manually
using `xxd -i <data file> > data_file.cc`

# Data Files in Examples

In order to clean up examples, make test inputs easier to understand, and
include TFLite models directly, TFLM has moved to generating the cc and header
files during the build process using a python script which `make` and `bazel`
call. To include data files in an example, generator inputs should be supplied
to the `microlite_test` call in the example's Makefile and `generate_cc_arrays`
should be used to create cc and header sources in the BUILD file.

For reference, see the
[Makefile](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/Makefile.inc)
and [BUILD](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/hello_world/BUILD)
files in the [hello_world
example](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/hello_world).

The generated cc and header files can be found in
`gen/<target>/genfiles/<path to example>`.
