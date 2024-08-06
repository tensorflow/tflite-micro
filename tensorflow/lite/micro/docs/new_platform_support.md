<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
   * [Porting to a new platform](#porting-to-a-new-platform)
      * [Step 1: Build TFLM Static Library with Reference Kernels](#step-1-build-tflm-static-library-with-reference-kernels)
      * [Step 2: Customize Logging and Timing Function for your Platform](#step-2-customize-logging-and-timing-function-for-your-platform)
      * [Step 3: Running the hello_world Example](#step-3-running-the-hello_world-example)
      * [Step 4: Building and Customizing Additional Examples](#step-4-building-and-customizing-additional-examples)
      * [Step 5: Integrating Optimized Kernel Implementations](#step-5-integrating-optimized-kernel-implementations)
   * [Advanced Integration Topics](#advanced-integration-topics)
   * [Getting Help](#getting-help)

<!-- Added by: advaitjain, at: Mon 04 Oct 2021 11:24:09 AM PDT -->

<!--te-->

# Porting to a new platform

At its core, TFLM is a portable library that can be used on a variety of target
hardware to run inference on TfLite models.

Prior to integrating TFLM with a specific hardware involves tasks that is
outside the scope of the TFLM project, including:

 * Toolchain setup - TFLM requires support for C++17
 * Set up and installation of board-specific SDKs and IDEs
 * Compiler flags and Linker setup
 * Integrating peripherals such as cameras, microphones and accelerometers to
   provide the sensor inputs for the ML models.

In this guide we outline our recommended approach for integrating TFLM with a
new target hardware assuming that you have already set up a development and
debugging environment for you board independent of TLFLM.


## Step 1: Build TFLM Static Library with Reference Kernels

Use the TFLM project generation script to create a directory tree containing
only the sources that are necessary to build the code TFLM library.

```bash
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world \
  -e micro_speech \
  -e person_detection \
  /tmp/tflm-tree
```

This will create a folder that looks like the following at the top-level:
```bash
examples  LICENSE  tensorflow  third_party
```

All the code in the `tensorflow` and `third_party` folders can be compiled into
a single static library (for example `libtflm.a`) using your platform-specific
build system.

TFLM's third party dependencies are spearated out in case there is a need to
have shared libraries for the third party code to avoid symbol collisions.

Note that for IDEs, it might be sufficient to simply include the
folder created by the TFLM project generation script into the overall IDE tree.

## Step 2: Customize Logging and Timing Function for your Platform

Replace the following files with a version that is specific to your target
platform:

 * [debug\_log.cc](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/debug_log.cc)
 * [micro\_time.cc](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_time.cc)
 * [system\_setup.cc](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/system_setup.cc)

These can be placed anywhere in your directory tree. The only requirement is
that when linking TFLM into a binary, the implementations of the functions in
[debug\_log.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/debug_log.h),
[micro\_time.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_time.h)
and [system\_setup.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/system_setup.h)
can be found.

For example, the implementations of these functions for:
  * [Sparkfun Edge](https://github.com/advaitjain/tflite-micro-sparkfun-edge-examples/tree/120f68ace95ae3d66963977ac7754acd0c86540d/tensorflow/lite/micro/sparkfun_edge)
is the implementation of these functions for the Sparkfun Edge.


## Step 3: Running the hello\_world Example

Once you have completed step 2, you should be set up to run the `hello_world`
example and see the output over the UART.

```
cp -r /tmp/tflm-tree/examples/hello_world <path-to-platform-specific-hello-world>
```
The `hello_world` example should not need any customization and you should be
able to directly build and run it.

## Step 4: Building and Customizing Additional Examples

We recommend that you fork the [TFLM examples](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples)
and then modify them as needed (to add support for peripherals etc.) to run on
your target platform.

## Step 5: Integrating Optimized Kernel Implementations

TFLM has optimized kernel implementations for a variety of targets that are in
sub-folders of the [kernels directory](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels).

It is possible to use the project generation script to create a tree with these
optimized kernel implementations (and associated third party dependencies).

For example:
```
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world -e micro_speech -e person_detection \
  --makefile_options="TARGET=cortex_m_generic OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET_ARCH=project_generation" \
  /tmp/tflm-cmsis
```

will create an output tree with all the sources and headers needed to use the
optimized [cmsis\_nn kernels](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/kernels/cmsis_nn) for Cortex-M platforms.


# Advanced Integration Topics

In order to have tighter coupling between your platform-specific TFLM
integration and the upstream TFLM repository, you might want to consider the
following:

 1. Set up a GitHub repository for your platform
 1. Nightly sync between TFLM and your platform-specific GitHub repository
 1. Using GitHub actions for CI

For some pointers on how to set this up, we refer you to the GitHub repositories
that integrated TFLM for the:
 * [Arduino](https://github.com/tensorflow/tflite-micro-arduino-examples): supported by the TFLM team
 * [Sparkfun Edge](https://github.com/advaitjain/tflite-micro-sparkfun-edge-examples): for demonstration purposes only, not officially supported.

Once you are set up with continuous integration and the ability to integrate
newer versions of TFLM with your platform, feel free to add a build badge to
TFLM's [Community Supported TFLM Examples](https://github.com/tensorflow/tflite-micro#community-supported-tflm-examples).

# Getting Help

[Here are some ways](https://github.com/tensorflow/tflite-micro#getting-help) that you can
reach out to get help.

