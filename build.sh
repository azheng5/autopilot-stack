#!/bin/bash

# IN PROGRESS

BUILD_DIR="build"

# cmake -G "Unix Makefiles" -B "$BUILD_DIR"
cmake -B "$BUILD_DIR"
cmake --build build

# Execute
# ./build <insert name of file> <arg1> <arg2>