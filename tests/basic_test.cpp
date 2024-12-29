#include <iostream>
#include "games/tic_tac_toe.h"

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
    std::cout << "Valid actions after moves: ";
    for (int action : state.GetValidActions()) {
        std::cout << action << " ";
    }
    std::cout << std::endl;

    // Check if the game is terminal
    std::cout << "Is terminal: " << state.IsTerminal() << std::endl;

    // Evaluate the current state
    std::cout << "State evaluation: " << state.Evaluate() << std::endl;

    return 0;
}