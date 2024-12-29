#include <iostream>
#include "games/tic_tac_toe.h"
#include "mcts/mcts.h"

int main() {
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(1000, state_factory);

    // Simulate a game using MCTS
    while (!mcts.IsTerminal(mcts.GetRoot())) {
        mcts.Search();
        int action = mcts.GetBestAction();
        
        // Debugging output
        std::cout << "MCTS selected action: " << action << std::endl;
        std::cout << "Current state before action:" << std::endl;
        mcts.GetRoot()->GetState()->Print();  // Assuming Print() is a method to display the state

        mcts.GetRoot()->GetState()->ApplyAction(action);

        // Update the root node to reflect the new state
        auto new_state = mcts.GetRoot()->GetState()->Clone();
        mcts.GetRoot() = std::make_shared<Node>(
            std::weak_ptr<Node>(),
            1 - mcts.GetRoot()->GetPlayerToMove(),
            action,
            std::move(new_state)
        );

        std::cout << "Current state after action:" << std::endl;
        mcts.GetRoot()->GetState()->Print();
    }

    std::cout << "Final state evaluation: " << mcts.GetRoot()->GetState()->Evaluate() << std::endl;
    std::cout << "Is terminal: " << mcts.IsTerminal(mcts.GetRoot()) << std::endl;

    return 0;
} 