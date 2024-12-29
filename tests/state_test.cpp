#include <iostream>
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
int main() {
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
    Logger::Log(LogLevel::INFO, "Valid actions after moves: ");
    for (int action : state.GetValidActions()) {
        Logger::Log(LogLevel::INFO, std::to_string(action) + " ");
    }
    std::cout << std::endl;

    // Check if the game is terminal
    Logger::Log(LogLevel::INFO, "Is terminal: " + std::to_string(state.IsTerminal()));

    // Evaluate the current state
    Logger::Log(LogLevel::INFO, "State evaluation: " + std::to_string(state.Evaluate()));

    return 0;
}