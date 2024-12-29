#!/bin/bash

# ============= build aarch64 release ===========
# build directory
BuildDir="build-android-aarch64-Release"
rm -rf $BuildDir
mkdir -p $BuildDir

cd $BuildDir

cmake -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME_r20b/build/cmake/android.toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_PLATFORM=android-29 \
      -DANDROID_STL=c++_shared \
      ..

make -j4
cd ..

