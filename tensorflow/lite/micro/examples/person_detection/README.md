# Person detection example

This example shows how you can use Tensorflow Lite to run a 250 kilobyte neural
network to recognize people in images.

## Table of contents

-   [Run the tests on a development machine](#run-the-tests-on-a-development-machine)
-   [Training your own model](#training-your-own-model)
-   [Running on ARC](#running-on-ARC)


## Run the tests on a development machine

```
make -f tensorflow/lite/micro/tools/make/Makefile third_party_downloads
make -f tensorflow/lite/micro/tools/make/Makefile test_person_detection_test
```

You should see a series of files get compiled, followed by some logging output
from a test, which should conclude with `~~~ALL TESTS PASSED~~~`. If you see
this, it means that a small program has been built and run that loads a trained
TensorFlow model, runs some example images through it, and got the expected
outputs. This particular test runs images with a and without a person in them,
and checks that the network correctly identifies them.

To understand how TensorFlow Lite does this, you can look at
[person_detection_test.cc](person_detection_test.cc).

## Training your own model

You can train your own model with some easy-to-use scripts. See
[training_a_model.md](training_a_model.md) for instructions.

## Running on ARC

### **Deploy on ARC EMSDP**
The following instructions will help you to build and deploy this example to
[ARC EM SDP](https://www.synopsys.com/dw/ipdir.php?ds=arc-em-software-development-platform)
board. General information and instructions on using the board with TensorFlow
Lite Micro can be found in the common
[ARC targets description](/tensorflow/lite/micro/tools/make/targets/arc/README.md).

This example uses asymmetric int8 quantization and can therefore leverage
optimized int8 kernels from the embARC MLI library

The ARC EM SDP board contains a rich set of extension interfaces. You can choose
any compatible camera and modify
[image_provider.cc](/tensorflow/lite/micro/examples/person_detection/image_provider.cc)
file accordingly to use input from your specific camera. By default, results of
running this example are printed to the console. If you would like to instead
implement some target-specific actions, you need to modify
[detection_responder.cc](/tensorflow/lite/micro/examples/person_detection/detection_responder.cc)
accordingly.

The reference implementations of these files are used by default on the EM SDP.

### Initial setup

Follow the instructions on the
[ARC EM SDP Initial Setup](/tensorflow/lite/micro/tools/make/targets/arc/README.md#ARC-EM-Software-Development-Platform-ARC-EM-SDP)
to get and install all required tools for work with ARC EM SDP.

### Generate Example Project

The example project for ARC EM SDP platform can be generated with the following
command:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_emsdp ARC_TAGS=reduce_codesize \
OPTIMIZED_KERNEL_DIR=arc_mli \
generate_person_detection_int8_make_project
```

Note that `ARC_TAGS=reduce_codesize` applies example specific changes of code to
reduce total size of application. It can be omitted.

### Build and Run Example

For more detailed information on building and running examples see the
appropriate sections of general descriptions of the
[ARC EM SDP usage with TensorFlow Lite Micro (TFLM)](/tensorflow/lite/micro/tools/make/targets/arc/README.md#ARC-EM-Software-Development-Platform-ARC-EM-SDP).
In the directory with generated project you can also find a
*README_ARC_EMSDP.md* file with instructions and options on building and
running. Here we only briefly mention main steps which are typically enough to
get it started.

1.  You need to
    [connect the board](/tensorflow/lite/micro/tools/make/targets/arc/README.md#connect-the-board)
    and open an serial connection.

2.  Go to the generated example project directory.

    ```
    cd tensorflow/lite/micro/tools/make/gen/arc_emsdp_arc_default/prj/person_detection_int8/make
    ```

3.  Build the example using

    ```
    make app
    ```

4.  To generate artefacts for self-boot of example from the board use

    ```
    make flash
    ```

5.  To run application from the board using microSD card:

    *   Copy the content of the created /bin folder into the root of microSD
        card. Note that the card must be formatted as FAT32 with default cluster
        size (but less than 32 Kbytes)
    *   Plug in the microSD card into the J11 connector.
    *   Push the RST button. If a red LED is lit beside RST button, push the CFG
        button.
    *   Type or copy next commands one-by-another into serial terminal: `setenv
        loadaddr 0x10800000 setenv bootfile app.elf setenv bootdelay 1 setenv
        bootcmd fatload mmc 0 \$\{loadaddr\} \$\{bootfile\} \&\& bootelf
        saveenv`
    *   Push the RST button.

6.  If you have the MetaWare Debugger installed in your environment:

    *   To run application from the console using it type `make run`.
    *   To stop the execution type `Ctrl+C` in the console several times.

In both cases (step 5 and 6) you will see the application output in the serial
terminal.

### **Deploy on ARC VPX processor**

The [embARC MLI Library 2.0](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA) enables TFLM library and examples to be used with the ARC VPX processor. This is currently an experimental feature. General information and instructions on using embARC MLI Library 2.0 with TFLM can be found in the common [ARC targets description](/tensorflow/lite/micro/tools/make/targets/arc/README.md).

### Initial Setup

Follow the instructions in the [Custom ARC EM/HS/VPX Platform](/tensorflow/lite/micro/tools/make/targets/arc/README.md#Custom-ARC-EMHSVPX-Platform) section to get and install all the required tools for working with the ARC VPX Processor.

### Generate Example Project

The example project for ARC VPX platform can be generated with the following
command:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_custom\
ARC_TAGS=mli20_experimental \
BUILD_LIB_DIR=<path_to_buildlib> \
TCF_FILE=<path_to_tcf_file> \
LCF_FILE=<path_to_lcf_file> \
OPTIMIZED_KERNEL_DIR=arc_mli \
generate_person_detection_int8_make_project
```
TCF file for VPX Processor can be generated using tcfgen tool which is part of [MetaWare Development Toolkit](#MetaWare-Development-Toolkit). \
The following command can be used to generate TCF file to run applications on VPX Processor using nSIM Simulator:
```
tcfgen -o vpx5_integer_full.tcf -tcf=vpx5_integer_full -iccm_size=0x80000 -dccm_size=0x40000
```
VPX Processor configuration may require a custom run-time library specified using the BUILD_LIB_DIR option. Please, check MLI Library 2.0 [documentation](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA#build-configuration-options) for more details. 

### Build and Run Example

For more detailed information on building and running examples see the
appropriate sections of general descriptions of the
[Custom ARC EM/HS/VPX Platform](/tensorflow/lite/micro/tools/make/targets/arc/README.md#Custom-ARC-EMHSVPX-Platform).
In the directory with generated project you can also find a
*README_ARC.md* file with instructions and options on building and
running. Here we only briefly mention main steps which are typically enough to
get started.

1.  Go to the generated example project directory.

    ```
    cd tensorflow/lite/micro/tools/make/gen/vpx5_integer_full_mli20_arc_default/prj/person_detection_int8/make
    ```

2.  Build the example using

    ```
    make app
    ```

3.  To run application from the MetaWare Debugger installed in your environment:

    *   From the console, type `make run`.
    *   To stop the execution type `Ctrl+C` in the console several times.

In both cases (step 5 and 6) you will see the application output in the serial
terminal.

