cmake_minimum_required(VERSION 3.18)
project(MCTS)

# Set the C++ standard and compiler flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -g -fopenmp")

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
    src/training/training_manager.cpp
    src/arena/arena_manager.cpp
    src/benchmarking/benchmark_runner.cpp
)

# Link torch to the library and set include directories
target_link_libraries(mcts_lib PRIVATE "${TORCH_LIBRARIES}")
target_include_directories(mcts_lib PRIVATE ${CMAKE_SOURCE_DIR}/src)

# State Test
add_executable(StateTest tests/state_test.cpp)
target_link_libraries(StateTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(StateTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# MCTS Test
add_executable(MCTSTest tests/mcts_test.cpp)
target_link_libraries(MCTSTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(MCTSTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Network Test
add_executable(NetworkTest tests/network_test.cpp)
target_link_libraries(NetworkTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(NetworkTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Agent Test
add_executable(AgentTest tests/agent_test.cpp)
target_link_libraries(AgentTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(AgentTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Arena Test
add_executable(ArenaTest tests/arena_test.cpp)
target_link_libraries(ArenaTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(ArenaTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Training Test
add_executable(TrainingTest tests/training_test.cpp)
target_link_libraries(TrainingTest mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(TrainingTest PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Create training executable
add_executable(TrainTicTacToe 
    experiments/tic_tac_toe/train.cpp
)
target_link_libraries(TrainTicTacToe mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(TrainTicTacToe PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add play executable
add_executable(PlayTicTacToe 
    experiments/tic_tac_toe/play.cpp
)
target_link_libraries(PlayTicTacToe mcts_lib "${TORCH_LIBRARIES}")
target_include_directories(PlayTicTacToe PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Set output directories for all tests
set_target_properties(
    StateTest 
    MCTSTest 
    NetworkTest 
    AgentTest 
    ArenaTest 
    TrainingTest
    TrainTicTacToe
    PlayTicTacToe
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Debug information
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch directory: ${Torch_DIR}")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mcts_lib PUBLIC OpenMP::OpenMP_CXX)
endif()
