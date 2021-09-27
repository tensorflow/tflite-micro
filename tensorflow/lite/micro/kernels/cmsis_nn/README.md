<!-- mdformat off(b/169948621#comment2) -->

# Info
CMSIS-NN is a library containing kernel optimizations for Arm(R) Cortex(TM)-M
processors. To use CMSIS-NN optimized kernels instead of reference kernels, add
`OPTIMIZED_KERNEL_DIR=cmsis_nn` to the make command line. See examples below.

For more information about the optimizations, check out
[CMSIS-NN documentation](https://github.com/ARM-software/CMSIS_5/blob/develop/CMSIS/NN/README.md)

By default CMSIS-NN is built by code that is downloaded to the TFLM tree.
It also possible to build CMSIS-NN code from an external path by specifying CMSIS_PATH=<../path>.
As a third option CMSIS-NN can be provided manually as an external library.
The examples below will illustrate this.

# Example 1

A simple way to compile a binary with CMSIS-NN optimizations.

```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn \
TARGET=sparkfun_edge person_detection_int8_bin
```

# Example 2 - MBED

Using mbed you'll be able to compile for the many different targets supported by
mbed. Here's an example on how to do that. Start by generating an mbed project.

```
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
--makefile_options OPTIMIZED_KERNEL_DIR=cmsis_nn \
--examples person_detection person_detection_build
```

Go into the generated build folder (in this case person_detection_build) and setup mbed:

```
cd person_detection_build
mbed new .
```

Now type:

```
mbed compile -m DISCO_F746NG -t GCC_ARM -D CMSIS_NN
```

and that gives you a binary for the DISCO_F746NG with CMSIS-NN optimized
kernels.

# Example 3 - FVP based on Arm Corstone-300 software.
Building the kernel conv unit test.
See tensorflow/lite/micro/cortex_m_corstone_300/README.md for more info about the target.

Downloaded CMSIS-NN code is built:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

External CMSIS-NN code is built:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn CMSIS_PATH=<external/path/to/cmsis/> TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

External CMSIS-NN library is linked in:
```
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn CMSIS_NN_LIBS=<path/to/cmsis-nn.a> CMSIS_PATH=<path/to/cmsis/> TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 kernel_conv_test
```

Please note that performance and/or size might be affected when using and external CMSIS-NN library as different compiler options may have been used.

Also please note that if specifying CMSIS_NN_LIBS but not CMSIS_PATH, headers and system/startup code from the default download path of CMSIS would be used. So CMSIS_NN_LIBS and CMSIS_PATH should have the same base path and if not there will be a build error.
