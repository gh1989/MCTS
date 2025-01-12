#include <iostream>
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include <torch/torch.h>
#include <cassert>

void TestTicTacToeStateInitialization() {
    TicTacToeState state;

    // Test initial valid actions
    std::cout << "Initial valid actions: ";
    for (int action : state.GetValidActions()) {
        std::cout << action << " ";
    }
    std::cout << std::endl;

    // Apply some actions
    state.ApplyAction(0);  // X moves
    state.ApplyAction(1);  // O moves
    state.ApplyAction(3);  // X moves

    // Test valid actions after some moves
    Logger::Log(LogLevel::TEST, "Valid actions after moves: ");
    for (int action : state.GetValidActions()) {
        Logger::Log(LogLevel::INFO, std::to_string(action) + " ");
    }
    std::cout << std::endl;

    // Check if the game is terminal
    Logger::Log(LogLevel::TEST, "Is terminal: " + std::to_string(state.IsTerminal()));

    // Evaluate the current state
    Logger::Log(LogLevel::TEST, "State evaluation: " + std::to_string(state.Evaluate()));
}

void TestTicTacToeStateToTensor() {
    Logger::Log(LogLevel::INFO, "Starting TicTacToe state to tensor test");
    
    TicTacToeState state;
    
    // Test initial state tensor
    torch::Tensor initial_tensor = state.ToTensor();
    assert(initial_tensor.sizes() == torch::IntArrayRef({3, 3, 3}));  // [channels, height, width]
    
    // Check that initial board is empty (all zeros in first two channels)
    assert(initial_tensor[0].sum().item<int>() == 0);  // X positions
    assert(initial_tensor[1].sum().item<int>() == 0);  // O positions
    
    // Check that it's X's turn (all ones in third channel)
    assert(initial_tensor[2].sum().item<int>() == 9);  // Turn channel
    
    // Make some moves and verify tensor
    state.ApplyAction(4);  // X plays center
    torch::Tensor after_move_tensor = state.ToTensor();
    
    // Check X's move
    assert(after_move_tensor[0][1][1].item<int>() == 1);  // Center position has X
    
    // Check it's O's turn (all zeros in third channel)
    assert(after_move_tensor[2].sum().item<int>() == 0);
    
    state.ApplyAction(0);  // O plays top-left
    torch::Tensor final_tensor = state.ToTensor();
    
    // Check O's move
    assert(final_tensor[1][0][0].item<int>() == 1);  // Top-left has O
    
    // Check it's X's turn again
    assert(final_tensor[2].sum().item<int>() == 9);
    
    Logger::Log(LogLevel::INFO, "TicTacToe state to tensor test passed");
}

int main() {
    TestTicTacToeStateInitialization();
    TestTicTacToeStateToTensor();
    return 0;
}