#!/bin/bash

# Create build directories if they don't exist
mkdir -p build/debug
mkdir -p build/release

# Function to handle build failures
handle_error() {
    echo "Error: Build failed"
    exit 1
}

# Parse command line arguments
BUILD_TYPE="Debug"
RUN_TESTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set build directory based on build type
BUILD_DIR="build/debug"
if [ "$BUILD_TYPE" = "Release" ]; then
    BUILD_DIR="build/release"
fi

# Configure and build
cd $BUILD_DIR || handle_error
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ../.. || handle_error
make -j$(nproc) || handle_error

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo "Running tests..."
    ./StateTest || handle_error
    ./MCTSTest || handle_error
    ./NetworkTest || handle_error
    ./AgentTest || handle_error
    ./ArenaTest || handle_error
    ./TrainingTest || handle_error
fi

echo "Build successful!" 