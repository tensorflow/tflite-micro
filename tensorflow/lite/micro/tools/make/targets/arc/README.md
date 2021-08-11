# Building TensorFlow Lite for Microcontrollers for Synopsys DesignWare ARC VPX and EM/HS Processors

## Maintainers

*   [dzakhar](https://github.com/dzakhar)
*   [JaccovG](https://github.com/JaccovG)
*   [gerbauz](https://github.com/gerbauz)

## Introduction

This document contains the general information on building and running
TensorFlow Lite Micro for targets based on the Synopsys ARC VPX and EM/HS Processors.

## Table of Contents

-   [Install the Synopsys DesignWare ARC MetaWare Development Toolkit](#install-the-synopsys-designware-arc-metaWare-development-toolkit)
-   [ARC EM Software Development Platform (ARC EM SDP)](#ARC-EM-Software-Development-Platform-ARC-EM-SDP)
-   [Using EmbARC MLI Library 2.0 (experimental feature)](#Using-EmbARC-MLI-Library-2.0-experimental-feature)
-   [Model Adaptation Tool (experimental feature)](#Model-Adaptation-Tool-experimental-feature)
-   [Custom ARC EM/HS/VPX Platform](#Custom-ARC-EMHSVPX-Platform)

## Install the Synopsys DesignWare ARC MetaWare Development Toolkit

The Synopsys DesignWare ARC MetaWare Development Toolkit (MWDT) is required to
build and run Tensorflow Lite Micro applications for all ARC VPX and EM/HS targets.

To license MWDT, please see further details
[here](https://www.synopsys.com/dw/ipdir.php?ds=sw_metaware)

To request an evaluation version of MWDT, please use the
[Synopsys Eval Portal](https://eval.synopsys.com/) and follow the link for the
MetaWare Development Toolkit (Important: Do not confuse this with MetaWare EV
Development Toolkit or MetaWare Lite options also available on this page)

Run the downloaded installer and follow the instructions to set up the toolchain
on your platform.

TensorFlow Lite for Microcontrollers builds are divided into two phases:
Application Project Generation and Application Project Building/Running. The
former phase requires \*nix environment while the latter does not.

For basic project generation targeting
[ARC EM Software Development Platform](#ARC-EM-Software-Development-Platform-ARC-EM-SDP),
MetaWare is NOT required for the Project Generation Phase. However, it is
required in case the following: - For project generation for custom (not EM SDP)
targets - To build microlib target library with all required TFLM objects for
external use

Please consider the above when choosing whether to install Windows or Linux or
both versions of MWDT

## ARC EM Software Development Platform (ARC EM SDP)

This section describes how to deploy on an
[ARC EM SDP board](https://www.synopsys.com/dw/ipdir.php?ds=arc-em-software-development-platform)

### Initial Setup

To use the EM SDP, you need the following hardware and software:

#### ARC EM SDP

More information on the platform, including ordering information, can be found
[here](https://www.synopsys.com/dw/ipdir.php?ds=arc-em-software-development-platform).

#### MetaWare Development Toolkit

See
[Install the Synopsys DesignWare ARC MetaWare Development Toolkit](#install-the-synopsys-designware-arc-metaWare-development-toolkit)
section for instructions on toolchain installation.

#### Digilent Adept 2 System Software Package

If you wish to use the MetaWare Debugger to debug your code, you need to also
install the Digilent Adept 2 software, which includes the necessary drivers for
connecting to the targets. This is available from official
[Digilent site](https://reference.digilentinc.com/reference/software/adept/start?redirect=1#software_downloads).
You should install the “System” component, and Runtime. Utilities and SDK are
NOT required.

Digilent installation is NOT required if you plan to deploy to EM SDP via the SD
card instead of using the debugger.

#### Make Tool

A `'make'` tool is required for both phases of deploying Tensorflow Lite Micro
applications on ARC EM SDP: 1. Application project generation 2. Working with
generated application (build and run)

For the first phase you need an environment and make tool compatible with
Tensorflow Lite for Micro build system. At the moment of this writing, this
requires make >=3.82 and a *nix-like environment which supports shell and native
commands for file manipulations. MWDT toolkit is not required for this phase.

For the second phase, requirements are less strict. The gmake version delivered
with MetaWare Development Toolkit is sufficient. There are no shell and *nix
command dependencies, so Windows can be used

#### Serial Terminal Emulation Application

The Debug UART port of the EM SDP is used to print application output. The USB
connection provides both the debug channel and RS232 transport. You can use any
terminal emulation program (like [PuTTY](https://www.putty.org/)) to view UART
output from the EM SDP.

#### microSD Card

If you want to self-boot your application (start it independently from a
debugger connection), you also need a microSD card with a minimum size of 512 MB
and a way to write to the card from your development host. Note that the card
must be formatted as FAT32 with default cluster size (but less than 32 Kbytes)

### Connect the Board

1.  Make sure Boot switches of the board (S3) are configured in the next way:

Switch # | Switch position
:------: | :-------------:
1        | Low (0)
2        | Low (0)
3        | High (1)
4        | Low (0)

1.  Connect the power supply included in the product package to the ARC EM SDP.
2.  Connect the USB cable to connector J10 on the ARC EM SDP (near the RST and
    CFG buttons) and to an available USB port on your development host.
3.  Determine the COM port assigned to the USB Serial Port (on Windows, using
    Device Manager is an easy way to do this)
4.  Execute the serial terminal application you installed in the previous step
    and open the serial connection with the early defined COM port (speed 115200
    baud; 8 bits; 1 stop bit; no parity).
5.  Push the CFG button on the board. After a few seconds you should see the
    boot log in the terminal which begins as follows:

```
U-Boot <Versioning info>

CPU:   ARC EM11D v5.0 at 40 MHz
Subsys:ARC Data Fusion IP Subsystem
Model: snps,emsdp
Board: ARC EM Software Development Platform v1.0
…
```

### Generate Application Project for ARC EM SDP

Before building an example or test application, you need to generate a TFLM
project for this application from TensorFlow sources and external dependencies.
To generate it for ARC EM SDP board you need to set `TARGET=arc_emsdp` on the
make command line. For instance, to build the Person Detect test application,
use a shell to execute the following command from the root directory of the
TensorFlow repo:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=arc_emsdp OPTIMIZED_KERNEL_DIR=arc_mli
generate_person_detection_test_int8_make_project
```

The application project will be generated into
*tensorflow/lite/micro/tools/make/gen/arc_emsdp_arc/prj/person_detection_test_int8/make*

Info on generating and building example applications for EM SDP
(*tensorflow/lite/micro/examples*) can be found in the appropriate readme file
placed in the same directory with the examples. In general, it’s the same
process which described in this Readme.

The
[embARC MLI Library](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli)
is used by default to speed up execution of some kernels for asymmetrically
quantized layers. Kernels which use MLI-based implementations are kept in the
*tensorflow/lite/micro/kernels/arc_mli* folder. For applications which may not
benefit from MLI library, the project can be generated without these
implementations by **removing** `OPTIMIZED_KERNEL_DIR=arc_mli` in the command line. This can reduce
code size when the optimized kernels are not required.

For more options on embARC MLI usage see
[kernels/arc_mli/README.md](/tensorflow/lite/micro/kernels/arc_mli/README.md).

### Build the Application

You may need to adjust the following commands in order to use the appropriate
make tool available in your environment (ie: `make` or `gmake`)

1.  Open command shell and change the working directory to the location which
    contains the generated project, as described in the previous section

2.  Clean previous build artifacts (optional)

    make clean

3.  Build application

    make app

### Run the Application with MetaWare Debugger on the nSim Simulator.

To run application from the console, use the following command:

```
   make run
```

If application runs in an infinite loop, type `Ctrl+C` several times to exit the
debugger.

To run the application in the GUI debugger, use the following command:

```
   make debug
```

You will see the application output in the same console where you ran it.

### Run the Application on the Board from the microSD Card

1.  Use the following command in the same command shell you used for building
    the application, as described in the previous step

```
    make flash
```

1.  Copy the content of the created *./bin* folder into the root of microSD
    card. Note that the card must be formatted as FAT32 with default cluster
    size (but less than 32 Kbytes)

2.  Plug in the microSD card into the J11 connector.

3.  Push the RST button. If a red LED is lit beside RST button, push the CFG
    button.

4.  Using serial terminal, create uboot environment file to automatically run
    the application on start-up. Type or copy next sequence of commands into
    serial terminal one-by-another:

```
   setenv loadaddr 0x10800000
   setenv bootfile app.elf
   setenv bootdelay 1
   setenv bootcmd fatload mmc 0 \$\{loadaddr\} \$\{bootfile\} \&\& bootelf
   saveenv
```

1.  Reset the board (see step 4 above)

You will see the application output in the serial terminal.

## Using EmbARC MLI Library 2.0 (experimental feature)

This section describes how to build TFLM using [embARC MLI Library 2.0](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA). 

The EmbARC MLI Library 2.0 can be used to build TFLM library and run applications (especially for VPX processors).

Because of difference in weights layout, TFLM models must be pre-adapted using a Model Adaptation Tool. For native TFLM examples (person detection, micro speech) Model Adaptation Tool is applied automatically when MLI 2.0 is used, so there is no need to run it maually.

To use the embARC MLI Library 2.0 in all cases (including native examples), you will also need extra dependencies for the Model Adaptation Tool. Please check the [Model Adaptation Tool](#​Model-Adaptation-Tool-experimental-​feature) section for more information.

To build TFLM using the embARC MLI Library 2.0, add the following tag to the command:
```
ARC_TAGS=mli20_experimental
```
Also, some of configurations may require custom BUILD_LIB. Please, check MLI Library 2.0 [documentation](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/tree/Release_2.0_EA#build-configuration-options) for more details. Following option can be added:
```
BUILD_LIB_DIR=<path_to_buildlib>
```
Example of command to build TFLM lib for VPX5:
```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_custom \
TCF=<path_to_tcf_file> \
BUILD_LIB_DIR=vpx5_integer_full \
ARC_TAGS=mli20_experimental microlite
```
## Model Adaptation Tool (experimental feature)

Models in TFLM format need to be pre-adapted before being used with MLI 2.0 due to differences in weights' tensor layout in some kernels. Adaptation is done automatically during TFLM project generation, but requires TensorFlow to be installed.

To use the Model Adaptation Tool, you need the following tools in addition to common requirments:
* [Python](https://www.python.org/downloads/) 3.7 or higher
* [TensorFlow for Python](https://www.tensorflow.org/install/pip) version 2.5 or higher

If you want to use your own model, exported from TensorFlow in **.tflite** or **.cc** format, you will need to adapt it manually using the Model Adaptation Tool from the current folder, using the following command:

```
python adaptation_tool.py <path_to_input_model_file> \
<path_to_adapted_model_file>
```

## Custom ARC EM/HS/VPX Platform

This section describes how to deploy on a Custom ARC VPX or EM/HS platform defined only by a TCF (Tool onfiguration File, created at CPU configuration time) and optional LCF (Linker Command File). In this case, the real hardware is unknown, and applications can be run only in the nSIM simulator included with the MetaWare toolkit.

VPX support is presented as an experimental feature of supporting embARC MLI Library version 2.0 and model adaptation. Read more about embARC MLI Library 2.0 support in the [related section](#Using-EmbARC-MLI-Library-2.0-experimental-feature).

### Initial Setup

To use a custom ARC EM/HS/VPX platform, you need the following : 
* Synopsys MetaWare
Development Toolkit version 2019.12 or higher (2021.06 or higher for MLI Library 2.0) 
* Make tool (make or gmake)
* CMake 3.18 or higher\
If you are using the [Model Adaptation Tool](#Model-Adaptation-Tool-experimental-feature), you will also need to install:
* [Python](https://www.python.org/downloads/) 3.7 or higher
* [TensorFlow for Python](https://www.tensorflow.org/install/pip) version 2.5 or higher

See
[Install the Synopsys DesignWare ARC MetaWare Development Toolkit](#install-the-synopsys-designware-arc-metaWare-development-toolkit)
section for instructions on toolchain installation. See
[MetaWare Development Toolkit](#MetaWare-Development-Toolkit) and
[Make Tool](#Make-Tool) sections for instructions on toolchain installation and
comments about make versions.


### Generate Application Project

Before building the application itself, you need to generate the project for
this application from TensorFlow sources and external dependencies. To generate
it for a custom TCF you need to set the following variables in the make command
line: * TARGET=arc_custom * TCF_FILE=<path to TCF file> * (optional)
LCF_FILE=<path to LCF file>

If you don’t supply an external LCF, the one embedded in the TCF will be used
instead

For instance, to build **Person Detection** test application, use the following
command from the root directory of the TensorFlow repo:

```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_custom \
OPTIMIZED_KERNEL_DIR=arc_mli \
TCF_FILE=<path_to_tcf_file> \
LCF_FILE=<path_to_lcf_file> \
generate_person_detection_int8_make_project
```
For MLI Library 2.0 (experimental feature):
```
make -f tensorflow/lite/micro/tools/make/Makefile \
TARGET=arc_custom \
OPTIMIZED_KERNEL_DIR=arc_mli \
ARC_TAGS=mli20_experimental \
BUILD_LIB_DIR=<path_to_buildlib> \
TCF_FILE=<path_to_tcf_file> \
LCF_FILE=<path_to_lcf_file> \
generate_person_detection_int8_make_project
```

The application project will be generated into
*tensorflow/lite/micro/tools/make/gen/<tcf_file_basename>_arc_default/prj/person_detection_test_int8/make*\
or\
  *tensorflow/lite/micro/tools/make/gen/<tcf_file_basename>_mli20_arc_default/prj/person_detection_test_int8/make* in case of usage MLI Library 2.0.

The
[embARC MLI Library](https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli)
is used by default to speed up execution of some kernels for asymmetrically
quantized layers. Kernels which use MLI-based implementations are kept in the
*tensorflow/lite/micro/kernels/arc_mli* folder. For applications which may not
benefit from MLI library, the project can be generated without these
implementations by **removing** `OPTIMIZED_KERNEL_DIR=arc_mli` in the command line. This can reduce
code size when the optimized kernels are not required.

For more options on embARC MLI usage see
[kernels/arc_mli/README.md](/tensorflow/lite/micro/kernels/arc_mli/README.md).

### Build the Application

You may need to adjust the following commands in order to use the appropriate
make tool available in your environment (ie: `make` or `gmake`)

1.  Open command shell and change the working directory to the location which
    contains the generated project, as described in the previous section

2.  Clean previous build artifacts (optional)

    make clean

3.  Build application

    make app

### Run the Application with MetaWare Debugger on the nSim Simulator.

To run application from the console, use the following command:

```
   make run
```

If application runs in an infinite loop, type `Ctrl+C` several times to exit the
debugger.

To run the application in the GUI debugger, use the following command:

```
   make debug
```

You will see the application output in the same console where you ran it.

## License

TensorFlow's code is covered by the Apache2 License included in the repository,
and third-party dependencies are covered by their respective licenses, in the
third_party folder of this package.
