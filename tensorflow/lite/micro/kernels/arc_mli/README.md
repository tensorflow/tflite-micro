# EmbARC MLI Library Based Optimizations of TensorFlow Lite Micro Kernels for ARC Platforms.

## Maintainers

*   [dzakhar](https://github.com/dzakhar)
*   [JaccovG](https://github.com/JaccovG)
*   [gerbauz](https://github.com/gerbauz)

## Introduction

This folder contains kernel implementations which use optimized
[embARC MLI Library](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli).
It allows acceleration of inference operations which use int8 (asymmetric
quantization).

## Usage

embARC MLI Library is used to speed up execution of some kernels for 
asymmetrically quantized layers and can be applied with the option `OPTIMIZED_KERNEL_DIR=arc_mli`.
This means that usual project generation for
ARC specific target implies usage of embARC MLI.

For example:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=arc_vpx OPTIMIZED_KERNEL_DIR=arc_mli generate_person_detection_int8_make_project
```

In case MLI implementation can’t be used, kernels in this folder fallback to
TFLM reference implementations. For applications which may not benefit from MLI
library, projects can be generated without these implementations **removing** `OPTIMIZED_KERNEL_DIR=arc_mli` in the command line, which can reduce overall code size:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=arc_vpx generate_person_detection_int8_make_project
```

For ARC EM SDP board, a pre-compiled MLI library is downloaded and used in the
application. For a custom target ARC-based platform, MLI sources are downloaded
and compiled during project generation phase. To build library from sources for
ARC EM SDP platform, add `BUILD_ARC_MLI=true` option to make command:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_emsdp \
OPTIMIZED_KERNEL_DIR=arc_mli \
BUILD_ARC_MLI=true \
generate_person_detection_int8_make_project
```
---
### Optional (experimental features):

TFLM can be built using [embARC MLI Library 2.0](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA) as an experimental feature.
To build TFLM using the embARC MLI Library 2.0, add the following tag to the command:
```
ARC_TAGS=mli20_experimental
```
In this case, generated projectes will be in <tcf_file_basename>_mli20_arc_default folder.

Some of configurations may require a custom run-time library specified using the BUILD_LIB_DIR option. Please, check MLI Library 2.0 [documentation](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA#build-configuration-options) for more details. The following option can be added:
```
BUILD_LIB_DIR=<path_to_buildlib>
```
---
If an application exclusively uses accelerated MLI kernel implementations, one
can strip out TFLM reference kernel implementations to reduce code size of
application. Build application with `MLI_ONLY=true` option in generated project
(after the project was built):

```
cd tensorflow/lite/micro/tools/make/gen/arc_vpx_arc_default/prj/person_detection_int8/make

make app MLI_ONLY=true
```

if you try this and application execution fails, then most probably MLI can’t be
used for some nodes and you need to revert to using TFLM reference kernels.

## Limitations

Currently, the MLI Library provides optimized implementation only for int8
(asymmetric) versions of the following kernels: 
1. Convolution 2D – Per axis
quantization only, `dilation_ratio==1` 
2. Depthwise Convolution 2D – Per axis
quantization only, `dilation_ratio==1` 
3. Average Pooling 
4. Max Pooling 
5. Fully Connected

## Scratch Buffers and Slicing

The following information applies only for ARC EM SDP, VPX and other targets with XY or VCCM
memory. embARC MLI uses specific optimizations which assumes node operands are
in XY, VCCM memory and/or DCCM (Data Closely Coupled Memory). As operands might be
quite big and may not fit in available XY or VCCM memory, special slicing logic is
applied which allows kernel calculations to be split into multiple parts. For
this reason, internal static buffers are allocated in these X, Y, VCCM and DCCM memory
banks and used to execute sub-calculations.

All this is performed automatically and invisible to the user. Half of the DCCM
memory bank and the full XY banks or 3/4 of VCCM bank are occupied for MLI specific needs.
If the user needs space in XY or VCCM memory for other tasks, these arrays can be reduced by
setting specific sizes. For this, add the following option to build command
replacing **<size[a|b|c]>** with required values:

**For EM:**
```
EXT_CFLAGS=”-DSCRATCH_MEM_Z_SIZE=<size_a> -DSCRATCH_MEM_X_SIZE=<size_b> -DSCRATCH_MEM_Y_SIZE=<size_c>”
```
**For VPX:**
```
EXT_CFLAGS=”-DSCRATCH_MEM_VEC_SIZE=<size_a>”
```

For example, to reduce sizes of arrays placed in DCCM and XCCM to 32k and 8k
respectively, use next command:

```
make app EXT_CFLAGS=”-DSCRATCH_MEM_Z_SIZE=32*1024 -DSCRATCH_MEM_X_SIZE=8*1024”
```

## License

TensorFlow's code is covered by the Apache2 License included in the repository,
and third party dependencies are covered by their respective licenses, in the
third_party folder of this package.
