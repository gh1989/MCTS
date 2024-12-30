#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run a test and check its result
run_test() {
    local test_name=$1
    echo "Running ${test_name}..."
    
    # Run the test and capture its exit code
    ./build/${test_name}
    local result=$?
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓ ${test_name} passed${NC}"
    else
        echo -e "${RED}✗ ${test_name} failed${NC}"
        return 1
    fi
}

# Make sure we're in the project root directory
cd "$(dirname "$0")/.."

# Build the project
echo "Building project..."
cd build && cmake .. && make
cd ..

# Array of test executables
tests=(
    "StateTest"
    "MCTSTest"
    "NetworkTest"
    "AgentTest"
    "ArenaTest"
    "TrainingTest"
)

# Counter for failed tests
failed=0

# Run each test
for test in "${tests[@]}"; do
    if ! run_test "$test"; then
        ((failed++))
    fi
    echo # Empty line for readability
done

# Print summary
total=${#tests[@]}
passed=$((total - failed))
echo "Test Summary: ${passed}/${total} tests passed"

# Exit with failure if any tests failed
[ $failed -eq 0 ] 