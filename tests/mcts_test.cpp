#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include "mcts/mcts.h"

void TestMCTSTreeConstruction() {
    Logger::Log(LogLevel::TEST, "Starting MCTS tree construction test.");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(10, state_factory);

    // Run the MCTS search
    mcts.Search();

    // Check the root node
    auto root = mcts.GetRoot();
    Logger::Log(LogLevel::TEST, "Root visit count: " + std::to_string(root->GetVisitCount()));
    Logger::Log(LogLevel::TEST, "Root children count: " + std::to_string(root->GetChildren().size()));

    // Check children of the root node
    for (const auto& child : root->GetChildren()) {
        Logger::Log(LogLevel::TEST, "Child action: " + std::to_string(child->GetAction()) +
                                     ", Visit count: " + std::to_string(child->GetVisitCount()));
    }

    // Verify that the tree has been expanded
    if (root->GetChildren().empty()) {
        Logger::Log(LogLevel::ERROR, "Error: Tree was not expanded.");
    } else {
        Logger::Log(LogLevel::TEST, "Tree expansion verified.");
    }
}

void TestMCTSTreeExpansion() {
auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(10, state_factory);

    // Simulate a game using MCTS
    while (!mcts.IsTerminal(mcts.GetRoot())) {
        mcts.Search();
        mcts.GetRoot()->PrintTree(mcts.GetRoot());
        int action = mcts.GetBestAction();
        
        // Debugging output
        Logger::Log(LogLevel::TEST, "MCTS selected action: " + std::to_string(action));
        mcts.GetRoot()->GetState()->Print();

        mcts.GetRoot()->GetState()->ApplyAction(action);

        // Update the root node to reflect the new state
        auto new_state = mcts.GetRoot()->GetState()->Clone();
        auto new_root = std::make_shared<Node>(
            std::weak_ptr<Node>(),
            1 - mcts.GetRoot()->GetPlayerToMove(),
            action,
            std::move(new_state)
        );
        mcts.SetRoot(new_root);

        mcts.GetRoot()->GetState()->Print();
    }

    Logger::Log(LogLevel::TEST, "Final state evaluation: " + std::to_string(mcts.GetRoot()->GetState()->Evaluate()));
    Logger::Log(LogLevel::TEST, "Is terminal: " + std::to_string(mcts.IsTerminal(mcts.GetRoot())));
}

// Function to recursively check the tree structure
bool CheckTreeConsistency(const std::shared_ptr<Node>& node) {
    if (!node) return true;

    // Check children recursively
    for (const auto& child : node->GetChildren()) {
        if (!CheckTreeConsistency(child)) {
            return false;
        }
    }

    // In MCTS, a node's visit count should be >= sum of child visits
    // because the node is visited during selection before expansion
    int total_child_visits = 0;
    for (const auto& child : node->GetChildren()) {
        total_child_visits += child->GetVisitCount();
    }
    
    if (node->GetVisitCount() < total_child_visits) {
        Logger::Log(LogLevel::TEST, "Invalid visit counts at node with action: " 
            + std::to_string(node->GetAction()) 
            + " Node visits: " + std::to_string(node->GetVisitCount())
            + " Sum of child visits: " + std::to_string(total_child_visits));
        return false;
    }

    return true;
}

void TestMCTSTreeExpansionConsistency() {
    Logger::Log(LogLevel::TEST, "Starting MCTS tree expansion consistency test.");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(1, state_factory);

    // Run the MCTS search
    mcts.Search();

    // Check the tree consistency
    if (CheckTreeConsistency(mcts.GetRoot())) {
        Logger::Log(LogLevel::TEST, "Tree expansion consistency test passed.");
    } else {
        Logger::Log(LogLevel::ERROR, "Tree expansion consistency test failed.");
    }
}

// Function to run a single simulation and log the details
void TestSingleSimulation() {
    Logger::Log(LogLevel::TEST, "Starting single simulation test.");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(10, state_factory);  // Set simulation count to 1 for a single simulation

    // Run a single MCTS search iteration
    std::shared_ptr<Node> selected = mcts.Select(mcts.GetRoot());
    std::shared_ptr<Node> expanded = mcts.Expand(selected);
    double value = mcts.Simulate(expanded ? expanded : selected);
    mcts.Backpropagate(expanded ? expanded : selected, value);

    // Log the visit counts and actions
    std::shared_ptr<Node> current = mcts.GetRoot();
    while (current) {
        Logger::Log(LogLevel::TEST, "Node action: " + std::to_string(current->GetAction()) +
                                    ", Visit count: " + std::to_string(current->GetVisitCount()) +
                                    ", Total value: " + std::to_string(current->GetTotalValue()));
        if (current->GetChildren().empty()) break;
        current = current->GetChildren().front();  // Move to the first child for simplicity
    }
}

int main() {
    TestMCTSTreeConstruction();
    TestMCTSTreeExpansion();
    TestMCTSTreeExpansionConsistency();
    TestSingleSimulation(); 
    return 0;
} 