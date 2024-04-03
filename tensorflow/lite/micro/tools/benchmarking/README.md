# Generic Benchmarking Tool build/run instructions
This tool can be used to benchmark any TfLite format model.  The tool can be
compiled in one of two ways:
1. Such that it takes command line arguments, allowing the path to the model
file to be specified as a program argument
2. With a model compiled into the tool, allowing use in any simulator or on
any hardware platform

Building the tool with the model compiled in uses two additional Makefile
variables:
* `BENCHMARK_MODEL_PATH`: the path to the TfLite format model file.  This
can be a relative or absolute path.  This variable is required.
* `BENCHMARK_ARENA_SIZE`: the size of the TFLM interpreter arena, in bytes.
This variable is optional.

## Tested, working targets
* x86
* cortex_m_qemu (no timing data)
* Xtensa
* cortex_m_corstone_300

## Tested, non-working targets
* none currently

## Build and run for x86
Build for command line arguments:
```
make -f tensorflow/lite/micro/tools/make/Makefile tflm_benchmark -j$(nproc)
```
Run with command line arguments:
```
gen/linux_x86_64_default/bin/tflm_benchmark tensorflow/lite/micro/models/person_detect.tflite
```

Build with model compiled into tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile tflm_benchmark -j$(nproc) BENCHMARK_MODEL_PATH=tensorflow/lite/micro/models/person_detect.tflite BENCHMARK_ARENA_SIZE=`expr 100 \* 1024`
```
Run with model compiled into tool:
```
gen/linux_x86_64_default/bin/tflm_benchmark
```

## Build and run for Xtensa
Build and run with model compiled into tool:
```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa TARGET_ARCH=vision_p6 OPTIMIZED_KERNEL_DIR=xtensa XTENSA_CORE=P6_200528 BUILD_TYPE=default run_tflm_benchmark -j$(nproc) BENCHMARK_MODEL_PATH=/tmp/keyword_scrambled.tflite BENCHMARK_ARENA_SIZE=`expr 50 \* 1024`
```
