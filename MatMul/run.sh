#!/bin/bash

# build
./build.sh

TarDir=/data/local/tmp/MatmulDemo

adb shell rm -rf $TarDir
adb shell mkdir $TarDir
adb push ./build-android-aarch64-Release/matmul $TarDir
adb shell "cd $TarDir; chmod 777 matmul; ./matmul"