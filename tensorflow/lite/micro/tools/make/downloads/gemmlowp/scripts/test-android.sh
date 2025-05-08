#!/bin/bash
# Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -z "$CXX" ]
then
  echo "please set the CXX environment variable to point to your native Android toolchain C++ compiler"
  exit 1
fi

default_cflags="-O3"

if [ "$#" -eq 0 ]
then
  echo "Usage: $0 files... [cflags...]"
  echo "All command-line parameters are passed along to the C++ compiler, so they can \
be either source files, or compiler flags."
  echo "Default cflags: $default_cflags"
  echo "Relies on the CXX environment variable to point to an Android C++ toolchain compiler."
  exit 1
fi

EXE=gemmlowp-android-binary

if [[ $CXX =~ .*aarch64.* ]]
then
  NEON_FLAGS=
else
  NEON_FLAGS="-mfpu=neon -mfloat-abi=softfp"
fi

$CXX \
 --std=c++11 \
 -Wall -Wextra -pedantic \
 -fPIE -pie $NEON_FLAGS \
 -lstdc++ -latomic \
 -I . -I .. \
 -o $EXE \
 -Wno-unused-variable -Wno-unused-parameter \
 $default_cflags \
 $*

if [ $? != 0 ]; then
  echo "build failed"
  exit 1
fi

adb root

if [ $? != 0 ]; then
  echo "$0: adb root failed"
  exit 1
fi

adb shell mkdir -p /data/local/tmp

if [ $? != 0 ]; then
  echo "$0: adb shell failed to mkdir /data/local/tmp"
  exit 1
fi

adb push $EXE /data/local/tmp

if [ $? != 0 ]; then
  echo "$0: adb push failed to write to /data/local/tmp"
  exit 1
fi

echo adb shell "/data/local/tmp/$EXE $TESTARGS"

adb shell "/data/local/tmp/$EXE $TESTARGS" | tee "log-$EXE"

if [ $? != 0 ]; then
  echo "$0: adb shell failed to run binary on device"
  exit 1
fi
