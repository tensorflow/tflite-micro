#!/bin/bash
if [ $TEST == "arm" ]; then
  ./android-ndk-${NDK_VERSION}/ndk-build
  android-wait-for-emulator
  # adb shell input keyevent 82 &
  adb push ./libs/* /data/local/tmp
  adb shell /data/local/tmp/benchmark
  adb shell /data/local/tmp/correctness_meta_gemm
  # too slow
  # adb shell /data/local/tmp/benchmark_meta_gemm
fi
if [ $TEST == "x86" ]; then
  make -f Makefile.travis unittest
fi
