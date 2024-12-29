cmake_minimum_required(VERSION 3.18)
project(MCTS)

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g")

# Set Torch directory and find package
set(Torch_DIR "/opt/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Create the main library
add_library(mcts_lib
    src/games/tic_tac_toe/tic_tac_toe.cpp
    src/common/logger.cpp
    src/mcts/mcts.cpp
)

# Link torch to the library and set include directories
target_link_libraries(mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(mcts_lib PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add the test executables
add_executable(StateTest tests/state_test.cpp)
target_link_libraries(StateTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(StateTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

add_executable(MCTSTest tests/mcts_test.cpp)
target_link_libraries(MCTSTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(MCTSTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Set output directories
set_target_properties(StateTest MCTSTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Debug information
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch directory: ${Torch_DIR}")
