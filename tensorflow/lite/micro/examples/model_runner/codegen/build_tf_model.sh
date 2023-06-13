#!/bin/bash

TENSORFLOW_ROOT=/home/cknorow/packages/tensorflow/

MODEL=$1

pushd $TENSORFLOW_ROOT

make -f tensorflow/lite/micro/tools/make/Makefile -j TARGET=zephyr_vexriscv  ${MODEL}_bin
#make -f tensorflow/lite/micro/tools/make/Makefile -j  $1_bin

TF_BUILD_DIR=$TENSORFLOW_ROOT/tensorflow/lite/micro/tools/make/gen/zephyr_vexriscv_x86_64

RENODE_DIR=~/packages/litex-vexriscv-tensorflow-lite-demo/
RENODE_SCRIPTS=$RENODE_DIR/renode/

cd $RENODE_RUNNER

rm -rf $RENODE_DIR/binaries/${MODEL}
mkdir  $RENODE_DIR/binaries/${MODEL}

cp ${TF_BUILD_DIR}/${MODEL}/build/zephyr/zephyr.elf $RENODE_DIR/binaries/${MODEL}/ 
cp ${TF_BUILD_DIR}/$1/build/zephyr/zephyr.bin $RENODE_DIR/binaries/${MODEL}/

cp ${TF_BUILD_DIR}/${MODEL}/build/zephyr/zephyr.elf $RENODE_DIR/binaries/
cp ${TF_BUILD_DIR}/${MODEL}/build/zephyr/zephyr.bin $RENODE_DIR/binaries/

popd
