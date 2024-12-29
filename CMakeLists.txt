cmake_minimum_required(VERSION 3.10)
project(MCTS)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(src)

# Add the test executable
add_executable(BasicTest
    tests/state_test.cpp
    src/games/tic_tac_toe/tic_tac_toe.cpp
    src/common/logger.cpp
    src/mcts/mcts.cpp
)

add_executable(MCTSTest
    tests/mcts_test.cpp
    src/games/tic_tac_toe/tic_tac_toe.cpp
    src/common/logger.cpp
    src/mcts/mcts.cpp
)

# Enable debugging
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
