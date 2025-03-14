<!-- mdformat off(b/169948621#comment2) -->

# General Info
CMSIS-NN is a library containing kernel optimizations for Arm(R) Cortex(R)-M
processors. To use CMSIS-NN optimized kernels instead of reference kernels, add
`OPTIMIZED_KERNEL_DIR=cmsis_nn` to the make command line. See examples below.

For more information about the optimizations, check out
[CMSIS-NN documentation](https://github.com/ARM-software/CMSIS-NN/blob/main/README.md),

# Specifying path to CMSIS-NN

By default CMSIS-NN is built by code that is downloaded to the TFLM tree.
It also possible to build CMSIS-NN code from an external path by specifying
CMSIS_PATH=<../path> and CMSIS_NN_PATH=<../path>. Note that both CMSIS_PATH and CMSIS_NN_PATH is needed
since CMSIS-NN has a dependency to CMSIS-Core. As a third option CMSIS-NN can be provided manually as an external library.
The examples below will illustrate this.

# Specifying path to Cortex_DFP

The Cortex_DFP path used can be specified using an additional flag `CORTEX_DFP_PATH=<path/to>cmsis/Cortex_DFP`.
Default is the Cortex_DFP contained in the downloaded CMSIS version.

## Example - FVP based on Arm Corstone-300 software.
In this example, the kernel conv unit test is built. For more information about
this specific target, check out the [Corstone-300 readme](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/cortex_m_corstone_300/README.md).

Downloaded CMSIS-NN code is built:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

External CMSIS-NN code is built:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn CMSIS_PATH=<external/path/to/cmsis/> CMSIS_NN_PATH=<external/path/to/cmsis-nn/>  TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

External CMSIS-NN library is linked in:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn CMSIS_NN_LIBS=<path/to/cmsis-nn.a> CMSIS_PATH=<path/to/cmsis/> TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

Please note that performance and/or size might be affected when using an
external CMSIS-NN library as different compiler options may have been used.

Also note that if specifying CMSIS_NN_LIBS but not CMSIS_PATH and or CMSIS_NN_PATH, headers and
system/startup code from the default downloaded path of CMSIS would be used.
So CMSIS_NN_LIBS, CMSIS_NN_PATH and CMSIS_PATH should have the same base path and if not there will be a build error.

# Build for speed or size
It is possible to build for speed or size. The size option may be required for a large model on an embedded system with limited memory. Where applicable, building for size would result in higher latency paired with a smaller scratch buffer, whereas building for speed would result in lower latency with a larger scratch buffer. Currently only transpose conv supports this.  See examples below.

## Example - building a static library with CMSIS-NN optimized kernels
More info on the target used in this example: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/cortex_m_generic/README.md

Bulding for speed (default):
Note that speed is default so if leaving out OPTIMIZE_KERNELS_FOR completely that will be the default.
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn OPTIMIZE_KERNELS_FOR=KERNELS_OPTIMIZED_FOR_SPEED microlite

```

Bulding for size:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m55 OPTIMIZED_KERNEL_DIR=cmsis_nn OPTIMIZE_KERNELS_FOR=KERNELS_OPTIMIZED_FOR_SIZE microlite

```
