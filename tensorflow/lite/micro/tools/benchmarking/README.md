# Generic Benchmarking Tool
This tool can be used to benchmark any TfLite format model.  The tool can be
compiled in one of two ways:
1. Such that it takes command line arguments, allowing the path to the model
file to be specified as a program argument
2. With a model compiled into the tool, allowing use in any simulator or on
any hardware platform

All tool output is prefaced with metadata.  The metadata consists of compiler
version and flags, and the target information supplied on the `make` command
line.  For some targets, version information for external libraries used with
optimized kernels is available.

If the model is compiled into the tool, additional model analysis information
is added to the metadata.  This includes data usage within the model, each model
subgraph and operation in inference execution order, and information on all
tensors in the model.

The tool will output a CRC32 of all input tensors, followed by the profiling
times for the pre-inference phase of the MicroInterpreter.  Next is the output
of the inference profiling times for each operator, and a summary total for
all inference operations.  Finally a CRC32 of all output tensors and the
MicroInterpreter arena memory usage are output.

# Generic Benchmarking Tool build/run instructions
Building the tool with the model compiled in uses two additional `make`
variables:
* `GENERIC_BENCHMARK_MODEL_PATH`: the path to the TfLite format model file.
The model path can be an abolute path, or relative to your local TFLM repository.
This variable is required.
* `GENERIC_BENCHMARK_ARENA_SIZE`: the size of the TFLM interpreter arena, in bytes.
This variable is optional.

## Tested targets
* x86
* cortex_m_qemu (no timing data)
* Xtensa (p6, hifi3, hifi5)
* cortex_m_corstone_300

## Use with compressed models
When the tool is used with compressed models, additional profiling timing will
be output.  This will consist of profiling timing for each tensor decompressed
during inference,
and a summary total.  While this profiling timing is output separately, the
timing for decompression is also included in the normal inference profiling
timing and summary total.

To use the tool with a compressed model, the `make` variables must include:
```
USE_TFLM_COMPRESSION=1
```

The tensor decompression operation can occur with an alternate destination
memory region.  This allows specialized memory to be used as the decompressed
data destination.  The tool supports a single alternate decompression region.
Use the following `make` variables to specify an alternate decompression region:
* `GENERIC_BENCHMARK_ALT_MEM_ATTR`: a C++ attribute specifying the alternate
memory as mapped through a linker script.
* `GENERIC_BENCHMARK_ALT_MEM_SIZE`: the alternate memory region size in bytes.

Both `make` variables are required (along with `USE_TFLM_COMPRESSION=1`) for the
tool to use the alternate decompression region.

An example build and run command line for Xtensa with alternate decompression memory:
```
make -f tensorflow/lite/micro/tools/make/Makefile  BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=compressed.tflite TARGET=xtensa TARGET_ARCH=hifi3 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=HIFI_190304_swupgrade USE_TFLM_COMPRESSION=1 GENERIC_BENCHMARK_ALT_MEM_ATTR='__attribute__\(\(section\(\".specialized_memory_region\"\)\)\)' GENERIC_BENCHMARK_ALT_MEM_SIZE=`expr 64 \* 1024`
```

For more information on model compression, please see the
[compression document](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/compression.md).

## Build and run for x86
Build for command line arguments:
```
make -f tensorflow/lite/micro/tools/make/Makefile tflm_benchmark -j$(nproc)
```
Run with command line arguments:
```
gen/linux_x86_64_default/bin/tflm_benchmark tensorflow/lite/micro/models/person_detect.tflite
```

Build and run with model compiled into tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite GENERIC_BENCHMARK_ARENA_SIZE=`expr 150 \* 1024`
```

## Build and run for Xtensa
Build and run with model compiled into tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa TARGET_ARCH=vision_p6 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=P6_200528 BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=/tmp/keyword_scrambled.tflite GENERIC_BENCHMARK_ARENA_SIZE=`expr 50 \* 1024`
```

Build and run with a compressed model compiled into the tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa TARGET_ARCH=vision_p6 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=P6_200528 BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=/tmp/keyword_scrambled.tflite GENERIC_BENCHMARK_ARENA_SIZE=`expr 50 \* 1024` USE_TFLM_COMPRESSION=1
```

## Build and run for Cortex-M using Corstone 300 simulator
Build and run with model compiled into tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile   TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m4   OPTIMIZED_KERNEL_DIR=cmsis_nn   BUILD_TYPE=default run_tflm_benchmark -j$(nproc) GENERIC_BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite GENERIC_BENCHMARK_ARENA_SIZE=`expr 150 \* 1024`
```

## Build and run using Bazel

This is only for the x86 command line argument build, and does not contain metadata:
```
bazel build tensorflow/lite/micro/tools/benchmarking:tflm_benchmark
bazel-bin/tensorflow/lite/micro/tools/benchmarking/tflm_benchmark tensorflow/lite/micro/models/person_detect.tflite
```

## Example output with metadata and built-in model layer information

This sample output is for Cortex-M using Corstone 300:
```
Configured arena size = 153600

--------------------
Compiled on:

Tue Dec 17 12:01:44 PM PST 2024
--------------------
Git SHA: aa47932ea602f72705cefe3fb9fc7fa2a651e205

Git status:

On branch your-test-branch

--------------------
C compiler: tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-gcc
Version:

arm-none-eabi-gcc (Arm GNU Toolchain 13.2.rel1 (Build arm-13.7)) 13.2.1 20231009
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Flags:

-Wimplicit-function-declaration -std=c17 -Werror -fno-unwind-tables 
-fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections -fmessage-length=0 
-DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -DCMSIS_NN 
-DKERNELS_OPTIMIZED_FOR_SPEED -mcpu=cortex-m4+nofp -mfpu=auto -DTF_LITE_MCU_DEBUG_LOG 
-mthumb -mfloat-abi=soft -funsigned-char -mlittle-endian -fomit-frame-pointer -MD -DARMCM4

C++ compiler: tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin/arm-none-eabi-g++
Version:

arm-none-eabi-g++ (Arm GNU Toolchain 13.2.rel1 (Build arm-13.7)) 13.2.1 20231009
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Flags:

-std=c++17 -fno-rtti -fno-exceptions -fno-threadsafe-statics -Wnon-virtual-dtor -Werror 
-fno-unwind-tables -fno-asynchronous-unwind-tables -ffunction-sections -fdata-sections 
-fmessage-length=0 -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON -Wsign-compare 
-Wdouble-promotion -Wunused-variable -Wunused-function -Wswitch -Wvla -Wall -Wextra 
-Wmissing-field-initializers -Wstrict-aliasing -Wno-unused-parameter -DCMSIS_NN 
-DKERNELS_OPTIMIZED_FOR_SPEED -mcpu=cortex-m4+nofp -mfpu=auto -DTF_LITE_MCU_DEBUG_LOG 
-mthumb -mfloat-abi=soft -funsigned-char -mlittle-endian -fomit-frame-pointer -MD 
-DARMCM4 -DCMSIS_DEVICE_ARM_CORTEX_M_XX_HEADER_FILE="ARMCM4.h" 
-DGENERIC_BENCHMARK_USING_BUILTIN_MODEL 
-DGENERIC_BENCHMARK_MODEL_HEADER_PATH="tensorflow/lite/micro/models/person_detect_model_da
ta.h" -DGENERIC_BENCHMARK_MODEL_NAME=person_detect 
-DGENERIC_BENCHMARK_TENSOR_ARENA_SIZE=153600

Optimization: kernel= -O2  core= -Os  third-party-kernel= -O2
--------------------
Target information:

TARGET=cortex_m_corstone_300
TARGET_ARCH=cortex-m4
OPTIMIZATION=cmsis_nn
BUILD_TYPE=default
--------------------
NN library download URLs:

http://github.com/ARM-software/CMSIS-NN/archive/22080c68d040c98139e6cb1549473e3149735f4d.z
ip

NN library MD5 checksums:

32aa69692541060a76b18bd5d2d98956
--------------------
Model SHA1:

bcafcaa99d2eaf089f0ca25d66f56a2177e93f76

Model analysis:

=== tensorflow/lite/micro/models/person_detect.tflite ===
Your TFLite model has '1' subgraph(s). In the subgraph description below,
T# represents the Tensor numbers. For example, in Subgraph#0, the DEPTHWISE_CONV_2D op 
takes
tensor #88 and tensor #0 and tensor #33 as input and produces tensor #34 as output.
Subgraph#0(T#88) -> [T#87]
  Op#0 DEPTHWISE_CONV_2D(T#88, T#0, T#33[3774, -107, -84394, -13908, 20697, ...]) -> 
[T#34]
  Op#1 DEPTHWISE_CONV_2D(T#34, T#9, T#52[31132, 28, 273, -2692, 7409, ...]) -> [T#51]
  Op#2 CONV_2D(T#51, T#10, T#53[10064, 1130, -13056, -30284, -23349, ...]) -> [T#54]
  Op#3 DEPTHWISE_CONV_2D(T#54, T#11, T#56[306, -158, 19181, -364, 6237, ...]) -> [T#55]
  Op#4 CONV_2D(T#55, T#12, T#57[-7649, 12287, -4433, 5851, -188, ...]) -> [T#58]
  Op#5 DEPTHWISE_CONV_2D(T#58, T#13, T#60[7297, -498, 263, -1975, 2260, ...]) -> [T#59]
  Op#6 CONV_2D(T#59, T#14, T#61[-4742, -4160, 6985, 8647, 29773, ...]) -> [T#62]
  Op#7 DEPTHWISE_CONV_2D(T#62, T#15, T#64[28588, 363, 27592, 22294, -4344, ...]) -> [T#63]
  Op#8 CONV_2D(T#63, T#16, T#65[12683, 36581, 6206, 1236, 15834, ...]) -> [T#66]
  Op#9 DEPTHWISE_CONV_2D(T#66, T#17, T#68[-6353, 9090, -30, -1019, -496, ...]) -> [T#67]
  Op#10 CONV_2D(T#67, T#18, T#69[3895, -6563, -8843, -2066, -1372, ...]) -> [T#70]
  Op#11 DEPTHWISE_CONV_2D(T#70, T#19, T#72[20437, -365, -2518, 20827, -904, ...]) -> 
[T#71]
  Op#12 CONV_2D(T#71, T#20, T#73[-10120, 9768, 3524, 3796, 6896, ...]) -> [T#74]
  Op#13 DEPTHWISE_CONV_2D(T#74, T#21, T#76[-3969, -1910, -2425, -114, 4456, ...]) -> 
[T#75]
  Op#14 CONV_2D(T#75, T#22, T#77[-13202, 13929, -4357, 19492, 1971, ...]) -> [T#78]
  Op#15 DEPTHWISE_CONV_2D(T#78, T#23, T#80[-6169, -10, -2788, 14420, -7457, ...]) -> 
[T#79]
  Op#16 CONV_2D(T#79, T#24, T#81[155, -3073, 291, -902, -9942, ...]) -> [T#82]
  Op#17 DEPTHWISE_CONV_2D(T#82, T#25, T#84[-2063, 10755, -12037, -6417, 2147, ...]) -> 
[T#83]
  Op#18 CONV_2D(T#83, T#26, T#85[-1872, -7549, 13994, 3191, -614, ...]) -> [T#86]
  Op#19 DEPTHWISE_CONV_2D(T#86, T#1, T#36[-6485, 294, 686, -6011, -5196, ...]) -> [T#35]
  Op#20 CONV_2D(T#35, T#2, T#37[7116, 8066, 11755, 11674, 9983, ...]) -> [T#38]
  Op#21 DEPTHWISE_CONV_2D(T#38, T#3, T#40[7735, 5235, 4334, -6485, 9397, ...]) -> [T#39]
  Op#22 CONV_2D(T#39, T#4, T#41[2947, 10152, -7865, -554, -13760, ...]) -> [T#42]
  Op#23 DEPTHWISE_CONV_2D(T#42, T#5, T#44[-4755, 7899, -488, -2954, 2990, ...]) -> [T#43]
  Op#24 CONV_2D(T#43, T#6, T#45[-6269, -22458, 13332, -16368, 4435, ...]) -> [T#46]
  Op#25 DEPTHWISE_CONV_2D(T#46, T#7, T#48[333, -4743, -310, -2471, 4804, ...]) -> [T#47]
  Op#26 CONV_2D(T#47, T#8, T#49[6677, -3593, 3754, 26316, -4761, ...]) -> [T#50]
  Op#27 AVERAGE_POOL_2D(T#50) -> [T#27]
  Op#28 CONV_2D(T#27, T#30, T#29[16267, -17079]) -> [T#28]
  Op#29 RESHAPE(T#28, T#32[1, 2]) -> [T#31]
  Op#30 SOFTMAX(T#31) -> [T#87]
Tensors of Subgraph#0
  T#0(MobilenetV1/Conv2d_0/weights/read) shape:[1, 3, 3, 8], type:INT8 RO 72 bytes, 
buffer: 68, data:[., y, ., g, ., ...]
  T#1(MobilenetV1/Conv2d_10_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 72, data:[W, ., d, ., ., ...]
  T#2(MobilenetV1/Conv2d_10_pointwise/weights/read) shape:[128, 1, 1, 128], type:INT8 RO 
16384 bytes, buffer: 14, data:[., ., 
, ., ., ...]
  T#3(MobilenetV1/Conv2d_11_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 13, data:[., `, ., :, ., ...]
  T#4(MobilenetV1/Conv2d_11_pointwise/weights/read) shape:[128, 1, 1, 128], type:INT8 RO 
16384 bytes, buffer: 12, data:[., ., ., ., ., ...]
  T#5(MobilenetV1/Conv2d_12_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 10, data:[z, ., ., ?, ., ...]
  T#6(MobilenetV1/Conv2d_12_pointwise/weights/read) shape:[256, 1, 1, 128], type:INT8 RO 
32768 bytes, buffer: 69, data:[/, ., ., ., #, ...]
  T#7(MobilenetV1/Conv2d_13_depthwise/depthwise_weights/read) shape:[1, 3, 3, 256], 
type:INT8 RO 2304 bytes, buffer: 7, data:[., ., w, ., ., ...]
  T#8(MobilenetV1/Conv2d_13_pointwise/weights/read) shape:[256, 1, 1, 256], type:INT8 RO 
65536 bytes, buffer: 5, data:[&, ., ., ., ., ...]
  T#9(MobilenetV1/Conv2d_1_depthwise/depthwise_weights/read) shape:[1, 3, 3, 8], 
type:INT8 RO 72 bytes, buffer: 60, data:[., ., ., ., ., ...]
  T#10(MobilenetV1/Conv2d_1_pointwise/weights/read) shape:[16, 1, 1, 8], type:INT8 RO 128 
bytes, buffer: 63, data:[., ., ., ., ., ...]
  T#11(MobilenetV1/Conv2d_2_depthwise/depthwise_weights/read) shape:[1, 3, 3, 16], 
type:INT8 RO 144 bytes, buffer: 58, data:[O, *, ., !, ., ...]
  T#12(MobilenetV1/Conv2d_2_pointwise/weights/read) shape:[32, 1, 1, 16], type:INT8 RO 
512 bytes, buffer: 61, data:[., 4, ., ., 8, ...]
  T#13(MobilenetV1/Conv2d_3_depthwise/depthwise_weights/read) shape:[1, 3, 3, 32], 
type:INT8 RO 288 bytes, buffer: 35, data:[., 1, ;, M, ., ...]
  T#14(MobilenetV1/Conv2d_3_pointwise/weights/read) shape:[32, 1, 1, 32], type:INT8 RO 
1024 bytes, buffer: 33, data:[., ., ., ., ., ...]
  T#15(MobilenetV1/Conv2d_4_depthwise/depthwise_weights/read) shape:[1, 3, 3, 32], 
type:INT8 RO 288 bytes, buffer: 32, data:[., ;, ., ., ., ...]
  T#16(MobilenetV1/Conv2d_4_pointwise/weights/read) shape:[64, 1, 1, 32], type:INT8 RO 
2048 bytes, buffer: 30, data:[., ., ., 5, ., ...]
  T#17(MobilenetV1/Conv2d_5_depthwise/depthwise_weights/read) shape:[1, 3, 3, 64], 
type:INT8 RO 576 bytes, buffer: 77, data:[G, ., ., ., ., ...]
  T#18(MobilenetV1/Conv2d_5_pointwise/weights/read) shape:[64, 1, 1, 64], type:INT8 RO 
4096 bytes, buffer: 28, data:[., 2, ., $, ., ...]
  T#19(MobilenetV1/Conv2d_6_depthwise/depthwise_weights/read) shape:[1, 3, 3, 64], 
type:INT8 RO 576 bytes, buffer: 27, data:[., 1, z, ., U, ...]
  T#20(MobilenetV1/Conv2d_6_pointwise/weights/read) shape:[128, 1, 1, 64], type:INT8 RO 
8192 bytes, buffer: 25, data:[5, ., ., ., V, ...]
  T#21(MobilenetV1/Conv2d_7_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 23, data:[., ., ., ., ., ...]
  T#22(MobilenetV1/Conv2d_7_pointwise/weights/read) shape:[128, 1, 1, 128], type:INT8 RO 
16384 bytes, buffer: 21, data:[., ., ., ., ., ...]
  T#23(MobilenetV1/Conv2d_8_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 71, data:[., ., ., ., Q, ...]
  T#24(MobilenetV1/Conv2d_8_pointwise/weights/read) shape:[128, 1, 1, 128], type:INT8 RO 
16384 bytes, buffer: 20, data:[@, ., 2, ., 8, ...]
  T#25(MobilenetV1/Conv2d_9_depthwise/depthwise_weights/read) shape:[1, 3, 3, 128], 
type:INT8 RO 1152 bytes, buffer: 80, data:[^, ., ~, ., ., ...]
  T#26(MobilenetV1/Conv2d_9_pointwise/weights/read) shape:[128, 1, 1, 128], type:INT8 RO 
16384 bytes, buffer: 16, data:[., .,  , ., %, ...]
  T#27(MobilenetV1/Logits/AvgPool_1a/AvgPool) shape:[1, 1, 1, 256], type:INT8
  T#28(MobilenetV1/Logits/Conv2d_1c_1x1/BiasAdd) shape:[1, 1, 1, 2], type:INT8
  T#29(MobilenetV1/Logits/Conv2d_1c_1x1/Conv2D_bias) shape:[2], type:INT32 RO 8 bytes, 
buffer: 2, data:[16267, -17079]
  T#30(MobilenetV1/Logits/Conv2d_1c_1x1/weights/read) shape:[2, 1, 1, 256], type:INT8 RO 
512 bytes, buffer: 3, data:[., %, ., ., ., ...]
  T#31(MobilenetV1/Logits/SpatialSqueeze) shape:[1, 2], type:INT8
  T#32(MobilenetV1/Logits/SpatialSqueeze_shape) shape:[2], type:INT32 RO 8 bytes, buffer: 
1, data:[1, 2]
  T#33(MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_bias) shape:[8], type:INT32 RO 32 bytes, 
buffer: 82, data:[3774, -107, -84394, -13908, 20697, ...]
  T#34(MobilenetV1/MobilenetV1/Conv2d_0/Relu6) shape:[1, 48, 48, 8], type:INT8
  T#35(MobilenetV1/MobilenetV1/Conv2d_10_depthwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#36(MobilenetV1/MobilenetV1/Conv2d_10_depthwise/depthwise_bias) shape:[128], 
type:INT32 RO 512 bytes, buffer: 22, data:[-6485, 294, 686, -6011, -5196, ...]
  T#37(MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Conv2D_bias) shape:[128], type:INT32 
RO 512 bytes, buffer: 70, data:[7116, 8066, 11755, 11674, 9983, ...]
  T#38(MobilenetV1/MobilenetV1/Conv2d_10_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#39(MobilenetV1/MobilenetV1/Conv2d_11_depthwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#40(MobilenetV1/MobilenetV1/Conv2d_11_depthwise/depthwise_bias) shape:[128], 
type:INT32 RO 512 bytes, buffer: 19, data:[7735, 5235, 4334, -6485, 9397, ...]
  T#41(MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Conv2D_bias) shape:[128], type:INT32 
RO 512 bytes, buffer: 11, data:[2947, 10152, -7865, -554, -13760, ...]
  T#42(MobilenetV1/MobilenetV1/Conv2d_11_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#43(MobilenetV1/MobilenetV1/Conv2d_12_depthwise/Relu6) shape:[1, 3, 3, 128], type:INT8
  T#44(MobilenetV1/MobilenetV1/Conv2d_12_depthwise/depthwise_bias) shape:[128], 
type:INT32 RO 512 bytes, buffer: 9, data:[-4755, 7899, -488, -2954, 2990, ...]
  T#45(MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Conv2D_bias) shape:[256], type:INT32 
RO 1024 bytes, buffer: 8, data:[-6269, -22458, 13332, -16368, 4435, ...]
  T#46(MobilenetV1/MobilenetV1/Conv2d_12_pointwise/Relu6) shape:[1, 3, 3, 256], type:INT8
  T#47(MobilenetV1/MobilenetV1/Conv2d_13_depthwise/Relu6) shape:[1, 3, 3, 256], type:INT8
  T#48(MobilenetV1/MobilenetV1/Conv2d_13_depthwise/depthwise_bias) shape:[256], 
type:INT32 RO 1024 bytes, buffer: 6, data:[333, -4743, -310, -2471, 4804, ...]
  T#49(MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Conv2D_bias) shape:[256], type:INT32 
RO 1024 bytes, buffer: 4, data:[6677, -3593, 3754, 26316, -4761, ...]
  T#50(MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6) shape:[1, 3, 3, 256], type:INT8
  T#51(MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6) shape:[1, 48, 48, 8], type:INT8
  T#52(MobilenetV1/MobilenetV1/Conv2d_1_depthwise/depthwise_bias) shape:[8], type:INT32 
RO 32 bytes, buffer: 56, data:[31132, 28, 273, -2692, 7409, ...]
  T#53(MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Conv2D_bias) shape:[16], type:INT32 RO 
64 bytes, buffer: 36, data:[10064, 1130, -13056, -30284, -23349, ...]
  T#54(MobilenetV1/MobilenetV1/Conv2d_1_pointwise/Relu6) shape:[1, 48, 48, 16], type:INT8
  T#55(MobilenetV1/MobilenetV1/Conv2d_2_depthwise/Relu6) shape:[1, 24, 24, 16], type:INT8
  T#56(MobilenetV1/MobilenetV1/Conv2d_2_depthwise/depthwise_bias) shape:[16], type:INT32 
RO 64 bytes, buffer: 48, data:[306, -158, 19181, -364, 6237, ...]
  T#57(MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Conv2D_bias) shape:[32], type:INT32 RO 
128 bytes, buffer: 62, data:[-7649, 12287, -4433, 5851, -188, ...]
  T#58(MobilenetV1/MobilenetV1/Conv2d_2_pointwise/Relu6) shape:[1, 24, 24, 32], type:INT8
  T#59(MobilenetV1/MobilenetV1/Conv2d_3_depthwise/Relu6) shape:[1, 24, 24, 32], type:INT8
  T#60(MobilenetV1/MobilenetV1/Conv2d_3_depthwise/depthwise_bias) shape:[32], type:INT32 
RO 128 bytes, buffer: 34, data:[7297, -498, 263, -1975, 2260, ...]
  T#61(MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Conv2D_bias) shape:[32], type:INT32 RO 
128 bytes, buffer: 59, data:[-4742, -4160, 6985, 8647, 29773, ...]
  T#62(MobilenetV1/MobilenetV1/Conv2d_3_pointwise/Relu6) shape:[1, 24, 24, 32], type:INT8
  T#63(MobilenetV1/MobilenetV1/Conv2d_4_depthwise/Relu6) shape:[1, 12, 12, 32], type:INT8
  T#64(MobilenetV1/MobilenetV1/Conv2d_4_depthwise/depthwise_bias) shape:[32], type:INT32 
RO 128 bytes, buffer: 31, data:[28588, 363, 27592, 22294, -4344, ...]
  T#65(MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Conv2D_bias) shape:[64], type:INT32 RO 
256 bytes, buffer: 76, data:[12683, 36581, 6206, 1236, 15834, ...]
  T#66(MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6) shape:[1, 12, 12, 64], type:INT8
  T#67(MobilenetV1/MobilenetV1/Conv2d_5_depthwise/Relu6) shape:[1, 12, 12, 64], type:INT8
  T#68(MobilenetV1/MobilenetV1/Conv2d_5_depthwise/depthwise_bias) shape:[64], type:INT32 
RO 256 bytes, buffer: 29, data:[-6353, 9090, -30, -1019, -496, ...]
  T#69(MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Conv2D_bias) shape:[64], type:INT32 RO 
256 bytes, buffer: 84, data:[3895, -6563, -8843, -2066, -1372, ...]
  T#70(MobilenetV1/MobilenetV1/Conv2d_5_pointwise/Relu6) shape:[1, 12, 12, 64], type:INT8
  T#71(MobilenetV1/MobilenetV1/Conv2d_6_depthwise/Relu6) shape:[1, 6, 6, 64], type:INT8
  T#72(MobilenetV1/MobilenetV1/Conv2d_6_depthwise/depthwise_bias) shape:[64], type:INT32 
RO 256 bytes, buffer: 26, data:[20437, -365, -2518, 20827, -904, ...]
  T#73(MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Conv2D_bias) shape:[128], type:INT32 RO 
512 bytes, buffer: 24, data:[-10120, 9768, 3524, 3796, 6896, ...]
  T#74(MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#75(MobilenetV1/MobilenetV1/Conv2d_7_depthwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#76(MobilenetV1/MobilenetV1/Conv2d_7_depthwise/depthwise_bias) shape:[128], type:INT32 
RO 512 bytes, buffer: 78, data:[-3969, -1910, -2425, -114, 4456, ...]
  T#77(MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Conv2D_bias) shape:[128], type:INT32 RO 
512 bytes, buffer: 83, data:[-13202, 13929, -4357, 19492, 1971, ...]
  T#78(MobilenetV1/MobilenetV1/Conv2d_7_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#79(MobilenetV1/MobilenetV1/Conv2d_8_depthwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#80(MobilenetV1/MobilenetV1/Conv2d_8_depthwise/depthwise_bias) shape:[128], type:INT32 
RO 512 bytes, buffer: 55, data:[-6169, -10, -2788, 14420, -7457, ...]
  T#81(MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Conv2D_bias) shape:[128], type:INT32 RO 
512 bytes, buffer: 18, data:[155, -3073, 291, -902, -9942, ...]
  T#82(MobilenetV1/MobilenetV1/Conv2d_8_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#83(MobilenetV1/MobilenetV1/Conv2d_9_depthwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#84(MobilenetV1/MobilenetV1/Conv2d_9_depthwise/depthwise_bias) shape:[128], type:INT32 
RO 512 bytes, buffer: 17, data:[-2063, 10755, -12037, -6417, 2147, ...]
  T#85(MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Conv2D_bias) shape:[128], type:INT32 RO 
512 bytes, buffer: 15, data:[-1872, -7549, 13994, 3191, -614, ...]
  T#86(MobilenetV1/MobilenetV1/Conv2d_9_pointwise/Relu6) shape:[1, 6, 6, 128], type:INT8
  T#87(MobilenetV1/Predictions/Reshape_1) shape:[1, 2], type:INT8
  T#88(input) shape:[1, 96, 96, 1], type:INT8
---------------------------------------------------------------
              Model size:     300568 bytes
    Non-data buffer size:      81640 bytes (27.16 %)
  Total data buffer size:     218928 bytes (72.84 %)
    (Zero value buffers):          0 bytes (00.00 %)
* Buffers of TFLite model are mostly used for constant tensors.
  And zero value buffers are buffers filled with zeros.
  Non-data buffers area are used to store operators, subgraphs and etc.
  You can find more details from 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs
--------------------
"Unique Tag","Total ticks across all events with that tag."
tflite::GetModel, 4
tflite::CreateOpResolver, 8090
tflite::RecordingMicroAllocator::Create, 40
tflite::MicroInterpreter instantiation, 59
tflite::MicroInterpreter::AllocateTensors, 363531
"total number of ticks", 371724

Input CRC32: 0x14F6A510

DEPTHWISE_CONV_2D took 224622 ticks (8 ms).
DEPTHWISE_CONV_2D took 175917 ticks (7 ms).
CONV_2D took 249561 ticks (9 ms).
DEPTHWISE_CONV_2D took 84958 ticks (3 ms).
CONV_2D took 145817 ticks (5 ms).
DEPTHWISE_CONV_2D took 164914 ticks (6 ms).
CONV_2D took 197283 ticks (7 ms).
DEPTHWISE_CONV_2D took 41304 ticks (1 ms).
CONV_2D took 99473 ticks (3 ms).
DEPTHWISE_CONV_2D took 79969 ticks (3 ms).
CONV_2D took 151505 ticks (6 ms).
DEPTHWISE_CONV_2D took 20053 ticks (0 ms).
CONV_2D took 78521 ticks (3 ms).
DEPTHWISE_CONV_2D took 38127 ticks (1 ms).
CONV_2D took 132863 ticks (5 ms).
DEPTHWISE_CONV_2D took 38127 ticks (1 ms).
CONV_2D took 132865 ticks (5 ms).
DEPTHWISE_CONV_2D took 38127 ticks (1 ms).
CONV_2D took 132859 ticks (5 ms).
DEPTHWISE_CONV_2D took 38127 ticks (1 ms).
CONV_2D took 132851 ticks (5 ms).
DEPTHWISE_CONV_2D took 38127 ticks (1 ms).
CONV_2D took 132853 ticks (5 ms).
DEPTHWISE_CONV_2D took 9585 ticks (0 ms).
CONV_2D took 78471 ticks (3 ms).
DEPTHWISE_CONV_2D took 17474 ticks (0 ms).
CONV_2D took 143615 ticks (5 ms).
AVERAGE_POOL_2D took 2229 ticks (0 ms).
CONV_2D took 386 ticks (0 ms).
RESHAPE took 28 ticks (0 ms).
SOFTMAX took 163 ticks (0 ms).

"Unique Tag","Total ticks across all events with that tag."
DEPTHWISE_CONV_2D, 1009435
CONV_2D, 1817013
AVERAGE_POOL_2D, 2269
RESHAPE, 87
SOFTMAX, 363694
"total number of ticks", 2820774

Output CRC32: 0xA4A6A6BE

[[ Table ]]: Arena
        Arena   Bytes   % Arena
        Total | 84436 |   100.00
NonPersistent | 55296 |    65.49
   Persistent | 29140 |    34.51

[[ Table ]]: Allocations
                  Allocation   Id    Used   Requested   Count   % Memory
            Eval tensor data |  0 |  1068 |      1068 |    89 |      1.26
      Persistent tensor data |  1 |    64 |        64 |     2 |      0.08
Persistent quantization data |  2 |    40 |        40 |     4 |      0.05
      Persistent buffer data |  3 | 25872 |     25704 |    90 |     30.64
 Tensor variable buffer data |  4 |     0 |         0 |     0 |      0.00
 Node and registration array |  5 |   992 |       992 |    31 |      1.17
              Operation data |  6 |     0 |         0 |     0 |      0.00

Application exit code: 0.

Info: /OSCI/SystemC: Simulation stopped by user.
[warning ][main@0][01 ns] Simulation stopped by user

--- FVP_MPS3_Corstone_SSE_300 statistics: -------------------------------------
Simulated time                          : 2.958458s
User time                               : 1.768731s
System time                             : 0.227094s
Wall time                               : 2.022361s
Performance index                       : 1.46
cpu0                                    :  36.57 MIPS (    73961463 Inst)
Memory highwater mark                   : 0x11935000 bytes ( 0.275 GB )
-------------------------------------------------------------------------------
```
