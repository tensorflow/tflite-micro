 <!-- mdformat off(b/169948621#comment2) -->

# Running a fixed virtual platform based on Arm(R) Corstone(TM)-300 software

This target makes use of a fixed virtual platform (FVP) based on Arm
Corstone-300 software.
- [More info about Arm Corstone-300](
https://developer.arm.com/ip-products/subsystem/corstone/corstone-300)
- [More info about FVPs](https://developer.arm.com/tools-and-software/simulation-models/fixed-virtual-platforms)

Building the Corstone-300 based target has the following dependencies:

-   [Arm Ethos-U Core Platform](https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-platform)
    -   Arm Ethos-U Core Platform provides the linker file as well as UART and retarget functions.
-   [CMSIS](https://github.com/ARM-software/CMSIS_6) + [CMSIS-Cortex_DFP](https://github.com/ARM-software/Cortex_DFP)
    -   CMSIS provides startup functionality, e.g. for setting up interrupt handlers and clock speed.
    -   See cmsis_download.sh for how these are downloaded relative to each other for the given examples and make targets.

Both these repositories are downloaded automatically by the build process in
TFLM.

# General build info

You can compile the Corstone-300 target for multiple Cortex-M CPUs. See below.

Required parameters:

-   ```TARGET```: cortex_m_corstone_300
-   ```TARGET_ARCH```: cortex-mXX. Replace XX with either of the options in the [Corstone-300 makefile](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/make/targets/cortex_m_corstone_300_makefile.inc)

# How to run

Note that Corstone-300 emulates a Cortex-M55 system, but it is backwards
compatible. This means one could run code compiled for e.g. a Cortex-M7.

Some examples:

```
make -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_network_tester_test
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_network_tester_test
make -f tensorflow/lite/micro/tools/make/Makefile CO_PROCESSOR=ethos_u OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_network_tester_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_network_tester_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 test_kernel_fully_connected_test
make -f tensorflow/lite/micro/tools/make/Makefile OPTIMIZED_KERNEL_DIR=cmsis_nn TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m7+fp test_kernel_fully_connected_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m3 test_kernel_fully_connected_test
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_corstone_300 TARGET_ARCH=cortex-m55 BUILD_TYPE=release_with_logs TOOLCHAIN=armclang test_network_tester_test
```
