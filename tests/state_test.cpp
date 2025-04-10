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

void TestTicTacToeStateTermination() {
    Logger::Log(LogLevel::TEST, "Starting TicTacToe termination test");
    
    TicTacToeState state;
    
    // Test 1: Horizontal win
    state.ApplyAction(0);  // X: top-left
    state.ApplyAction(3);  // O: middle-left
    state.ApplyAction(1);  // X: top-middle
    state.ApplyAction(4);  // O: center
    state.ApplyAction(2);  // X: top-right
    
    if (!state.IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Failed to detect horizontal win");
        state.Print();
        return;
    }
    
    // Test 2: Vertical win
    TicTacToeState state2;
    state2.ApplyAction(0);  // X: top-left
    state2.ApplyAction(1);  // O: top-middle
    state2.ApplyAction(3);  // X: middle-left
    state2.ApplyAction(4);  // O: center
    state2.ApplyAction(6);  // X: bottom-left
    
    if (!state2.IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Failed to detect vertical win");
        state2.Print();
        return;
    }
    
    // Test 3: Diagonal win
    TicTacToeState state3;
    state3.ApplyAction(0);  // X: top-left
    state3.ApplyAction(1);  // O: top-middle
    state3.ApplyAction(4);  // X: center
    state3.ApplyAction(3);  // O: middle-left
    state3.ApplyAction(8);  // X: bottom-right
    
    if (!state3.IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Failed to detect diagonal win");
        state3.Print();
        return;
    }
    
    // Test 4: Draw game
    TicTacToeState state4;
    // X O X
    // X O O
    // O X X
    state4.ApplyAction(0);  // X
    state4.ApplyAction(1);  // O
    state4.ApplyAction(2);  // X
    state4.ApplyAction(4);  // O
    state4.ApplyAction(3);  // X
    state4.ApplyAction(5);  // O
    state4.ApplyAction(7);  // X
    state4.ApplyAction(6);  // O
    state4.ApplyAction(8);  // X
    
    if (!state4.IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Failed to detect draw game");
        state4.Print();
        return;
    }
    
    // Test 5: Non-terminal state
    TicTacToeState state5;
    state5.ApplyAction(0);  // X: top-left
    state5.ApplyAction(4);  // O: center
    
    if (state5.IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Incorrectly detected terminal state");
        state5.Print();
        return;
    }
    
    Logger::Log(LogLevel::TEST, "TicTacToe termination test passed");
}

int main() {
    TestTicTacToeStateInitialization();
    TestTicTacToeStateToTensor();
    TestTicTacToeStateTermination();
    return 0;
}