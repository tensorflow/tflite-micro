# TFLM Compression Support

TFLM supports fixed width compression of const-tensors using lookup tables.
Const-tensors are typically those containing trained weights or biases, but can
be any tensor where the values are fixed within the model and unchanging.

Const-tensors are compressed to fixed-width bit-strings, and lookup tables are
added to the model schema for each tensor.

When accessing a compressed tensor, each kernel invokes a common decompression
method.  Each set of fixed-width bits in the tensor bit-string are used as
indices into the tensor lookup table.  The results of the lookup table operations
are placed into a scratch buffer representing the tensor decompressed data.

Decompression results in increased latency during inference.

# Model Schema and Metadata for Compression

TBD

# Supported Tensor Types

* FLOAT32, INT8, INT16, INT32, INT64, BOOL

# Supported Kernels

* FULLY_CONNECTED
* CONV_2D
* DEPTHWISE_CONV
* TRANSPOSE_CONV
* CONCATENATION
* ASSIGN_VARIABLE

Per-channel quantized tensor support is available for:
* CONV_2D
* DEPTHWISE_CONV
* TRANSPOSE_CONV

# Supported Platforms

* X86
* XTENSA
  * P6_VISION, HIFI_MINI, HIFI3, HIFI4, HIFI5

# How to Compress a Model

Compression works best when the targeted tensors in the model have been binned.
Binning of the model tensors will result in a change in model accuracy, but will
also allow for better control of the compression ratio.  For example, by binning
a tensor to just four values among the tensor elements, a fixed-width of two bits
can be used for each element.  This would result in nearly a four-fold decrease
in the tensor size.

First, align your binned model:
```
bazel run --cache_test_results=no --test_output=all -s  tensorflow/lite/micro/tools:tflite_flatbuffer_align -- binned_model.tflite binned_and_aligned.tflite
```

Next, compress the model, supplying as arguments the target tensors:
```
bazel run --cache_test_results=no --test_output=all -s  tensorflow/lite/micro/compression:compress -- binned_and_aligned.tflite compressed.tflite --tensors="56, 54, 53, 51, 50, 49, 48, 46, 45, 44, 43, 27, 40, 26, 38, 37, 25, 35, 24, 33, 32, 23, 30"
```

Then align the model:
```
bazel run --cache_test_results=no --test_output=all -s  tensorflow/lite/micro/tools:tflite_flatbuffer_align -- compressed.tflite compressed_and_aligned.tflite
```

# How to Run the Generic Benchmark Application

The Generic Benchmark Application can only be built using `make`.

The benchmark output includes a CRC32 of the output tensor(s) for comparison
within the same platform on which the benchmark is run.

## Without Compression

HIFI3 example:
```
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile  BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=binned_and_aligned.tflite TARGET=xtensa TARGET_ARCH=hifi3 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=HIFI_190304_swupgrade
```

The model path can be an abolute path, or relative to your local TFLM repository.

## With Compression

HIFI5 example:
```
make -f ${TENSORFLOW_ROOT}tensorflow/lite/micro/tools/make/Makefile  BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=compressed_and_aligned.tflite TARGET=xtensa TARGET_ARCH=hifi5 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=PRD_H5_RDO_07_01_2022 USE_TFLM_COMPRESSION=1
```

The model path can be an abolute path, or relative to your local TFLM repository.

