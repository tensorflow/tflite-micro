#!/bin/bash

rm -rf tensorflow/lite/micro/tools/make/gen/*

make -j -f tensorflow/lite/micro/tools/make/Makefile generate_model_runner_make_project
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=x86mingw64 generate_model_runner_make_project
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_a_generic TARGET_ARCH=cortex-a53+fp generate_model_runner_make_project TOOLCHAIN=arm-none-linux-gnueabihf GNU_INSTALL_ROOT=/usr/local/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_a_generic TARGET_ARCH=cortex-a72+fp generate_model_runner_make_project TOOLCHAIN=arm-none-linux-gnueabihf GNU_INSTALL_ROOT=/usr/local/gcc-arm-10.3-2021.07-x86_64-arm-none-linux-gnueabihf
make -j -f tensorflow/lite/micro/tools/make/Makefile TARGET=mplab_xc32 TARGET_ARCH=ATSAMD21G18A generate_model_runner_make_project TOOLCHAIN=xc32-  OPTIMIZED_KERNEL_DIR=cmsis_nn


pushd tensorflow/lite/micro/tools/make/gen/linux_x86_64_default/prj/model_runner/make/
if [ ! -f libtensorflow-microlite.a ]; then
    sed -i 's/-lm/-lm -fPIC/' Makefile
    sed -i 's/-fno-rtti/-fno-rtti -fPIC/' Makefile
    sed -i 's/-std=c11/-std=c11 -fPIC/' Makefile
    
    make lib -j 
    cp libtensorflow-microlite.a $SENSIML_SERVER_DEV_HOME/kbserver/codegen/templates/platforms/x86gcc_generic/libsensiml/libtensorflow-microlite.a
    exit
fi
popd


pushd tensorflow/lite/micro/tools/make/gen/x86mingw64_x86_64_default/prj/model_runner/make/
if [ ! -f libtensorflow-microlite.a ]; then
    docker run -it -w /tflite-micro -v $(pwd):/tflite-micro 358252950181.dkr.ecr.us-west-2.amazonaws.com/sml_x86mingw_generic:9.3 make -j lib 
    cp libtensorflow-microlite.a $SENSIML_SERVER_DEV_HOME/kbserver/codegen/templates/platforms/x86mingw64_generic/libsensiml/libtensorflow-microlite.a
fi
popd


pushd tensorflow/lite/micro/tools/make/gen/cortex_a_generic_cortex-a53+fp_default/prj/model_runner/make/
if [ ! -f libtensorflow-microlite.a ]; then
    docker run -it -w /tflite-micro -v $(pwd):/tflite-micro 358252950181.dkr.ecr.us-west-2.amazonaws.com/sensiml/arm_none_linux_gnueabihf_base:10.3.1 make -j lib 
    cp libtensorflow-microlite.a $SENSIML_SERVER_DEV_HOME/kbserver/codegen/templates/platforms/raspberry_pi/libsensiml/cortex-a53+fp/libtensorflow-microlite.a
fi
popd

pushd tensorflow/lite/micro/tools/make/gen/cortex_a_generic_cortex-a72+fp_default/prj/model_runner/make/
if [ ! -f libtensorflow-microlite.a ]; then
    docker run -it -w /tflite-micro -v $(pwd):/tflite-micro 358252950181.dkr.ecr.us-west-2.amazonaws.com/sensiml/arm_none_linux_gnueabihf_base:10.3.1 make -j lib 
    cp libtensorflow-microlite.a $SENSIML_SERVER_DEV_HOME/kbserver/codegen/templates/platforms/raspberry_pi/libsensiml/cortex-a72+fp/libtensorflow-microlite.a
fi
popd


pushd tensorflow/lite/micro/tools/make/gen/mplab_xc32_ATSAMD21G18A_default/prj/model_runner/make/
if [ ! -f libtensorflow-microlite.a ]; then
    docker run -it -w /tflite-micro -v $(pwd):/tflite-micro 358252950181.dkr.ecr.us-west-2.amazonaws.com/sml_microchip_xc32:4.00 make -j lib 
popd
fi


