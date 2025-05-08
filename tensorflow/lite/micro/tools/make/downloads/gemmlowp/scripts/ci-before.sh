#!/bin/bash
if [ $TEST == "arm" ]; then
  curl -L https://dl.google.com/android/repository/android-ndk-${NDK_VERSION}-linux-x86_64.zip -O
  unzip android-ndk-${NDK_VERSION}-linux-x86_64.zip 2> /dev/null > /dev/null
  echo no | android create avd --force -n test -t android-22 --abi armeabi-v7a
  emulator -avd test -no-audio -no-window &
fi
