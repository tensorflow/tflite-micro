# TFLite for Microcontrollers Benchmarks

These benchmarks are for measuring the performance of key models and workloads.
They are meant to be used as part of the model optimization process for a given
platform.

## Table of contents

-   [Keyword Benchmark](#keyword-benchmark)
-   [Person Detection Benchmark](#person-detection-benchmark)
-   [Run on x86](#run-on-x86)
-   [Run on Xtensa XPG Simulator](#run-on-xtensa-xpg-simulator)
-   [Run on Sparkfun Edge](#run-on-sparkfun-edge)
-   [Run on FVP based on Arm Corstone-300 software](#run-on-fvp-based-on-arm-corstone-300-software)

## Keyword benchmark

The keyword benchmark contains a model for keyword detection with scrambled
weights and biases.  This model is meant to test performance on a platform only.
Since the weights are scrambled, the output is meaningless. In order to validate
the accuracy of optimized kernels, please run the kernel tests.

## Person detection benchmark

The keyword benchmark provides a way to evaluate the performance of the 250KB
visual wakewords model.

## Run on x86

To run the keyword benchmark on x86, run

```
make -f tensorflow/lite/micro/tools/make/Makefile run_keyword_benchmark
```

To run the person detection benchmark on x86, run

```
make -f tensorflow/lite/micro/tools/make/Makefile run_person_detection_benchmark
```

## Run on Xtensa XPG Simulator

To run the keyword benchmark on the Xtensa XPG simulator, you will need a valid
Xtensa toolchain and license.  With these set up, run:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa OPTIMIZED_KERNEL_DIR=xtensa TARGET_ARCH=<target architecture> XTENSA_CORE=<xtensa core> run_keyword_benchmark -j18
```

## Run on Sparkfun Edge
The following instructions will help you build and deploy this benchmark on the
[SparkFun Edge development board](https://sparkfun.com/products/15170).


If you're new to using this board, we recommend walking through the
[AI on a microcontroller with TensorFlow Lite and SparkFun Edge](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow)
codelab to get an understanding of the workflow.

Build binary using

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=sparkfun_edge person_detection_benchmark_bin
```

Refer to flashing instructions in the [Person Detection Example](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/person_detection/README.md#running-on-sparkfun-edge).

## Run on FVP based on Arm Corstone-300 software

For more info about the Corstone-300 software see:
[tensorflow/lite/micro/cortex_m_corstone_300/README.md](../cortex_m_corstone_300/README.md).

Disclaimer: The FVP can not be used to measure CPU performance.
The results are not reliable, not even for relative measurements.
FVP may however be used for performance measurements when running on NPU and only NPU PMU numbers can be used. The NPU model is cycle accurate within approximately +-10%.

As an example, the person detect downloaded model will be optimized for Ethos-U. For more info see:
[tensorflow/lite/micro/kernels/ethos_u/README.md](../kernels/ethos_u/README.md).
And since it only makes sense to measure performance on the NPU, only the person detection benchmark should be run and only with Ethos-U enabled.
See also network tester example, where person detect model is used in the same way when Ethos-U is enabled:
[tensorflow/lite/micro/examples/network_tester/README.md](../examples/network_tester/README.md).

The person detect model is not an optimial model for Ethos-U since it quite small. Also note that only the NPU PMU cycles are logged even though the CPU is setting up the Ethos-U driver in each iteration.

To run the person detection benchmark on FVP with Ethos-U:

```
make -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 run_person_detection_benchmark
```
