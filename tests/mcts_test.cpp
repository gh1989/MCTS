#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include "mcts/mcts.h"
#include <cmath>
#include <numeric>

void TestMCTSTreeConstruction() {
    Logger::Log(LogLevel::TEST, "Starting MCTS tree construction test");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(100, state_factory);

    // Run the MCTS search
    mcts.Search();

    // Check the root node
    auto root = mcts.GetRoot();
    Logger::Log(LogLevel::TEST, "Root visit count: " + std::to_string(root->GetVisitCount()));
    Logger::Log(LogLevel::TEST, "Root children count: " + std::to_string(root->GetChildren().size()));

    // Verify root properties
    if (root->GetVisitCount() < 100) {
        Logger::Log(LogLevel::ERROR, "Root node has insufficient visits");
        return;
    }

    if (root->GetChildren().empty()) {
        Logger::Log(LogLevel::ERROR, "Root has no children");
        return;
    }

    Logger::Log(LogLevel::TEST, "Tree construction test passed");
}

void TestMCTSSearchBehavior() {
    Logger::Log(LogLevel::TEST, "Starting MCTS search behavior test");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    MCTS mcts(100, state_factory);

    // Run multiple searches and track statistics
    std::vector<int> actions;
    std::vector<double> visit_ratios;

    for (int i = 0; i < 5; i++) {
        mcts.Search();
        auto root = mcts.GetRoot();
        int action = mcts.GetBestAction();
        actions.push_back(action);
        
        // Calculate visit concentration
        int best_visits = 0;
        for (const auto& child : root->GetChildren()) {
            if (child->GetVisitCount() > best_visits) {
                best_visits = child->GetVisitCount();
            }
        }
        
        double visit_ratio = static_cast<double>(best_visits) / root->GetVisitCount();
        visit_ratios.push_back(visit_ratio);
        
        Logger::Log(LogLevel::DEBUG, "Search " + std::to_string(i) + 
                   ": Action=" + std::to_string(action) + 
                   ", Visit ratio=" + std::to_string(visit_ratio));
    }

    // Check for reasonable visit distribution
    double avg_ratio = std::accumulate(visit_ratios.begin(), visit_ratios.end(), 0.0) / visit_ratios.size();
    if (avg_ratio < 0.2 || avg_ratio > 0.8) {
        Logger::Log(LogLevel::ERROR, "Unusual visit distribution: " + std::to_string(avg_ratio));
        return;
    }

    Logger::Log(LogLevel::TEST, "Search behavior test passed");
}

void TestMCTSExploration() {
    Logger::Log(LogLevel::TEST, "Starting MCTS exploration test");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    
    // Test different exploration constants
    std::vector<double> exploration_constants = {0.5, 1.4142, 2.0};
    
    for (double c : exploration_constants) {
        MCTS mcts(100, state_factory, c);
        mcts.Search();
        
        auto root = mcts.GetRoot();
        std::vector<int> visit_counts;
        
        for (const auto& child : root->GetChildren()) {
            visit_counts.push_back(child->GetVisitCount());
        }
        
        // Calculate visit distribution statistics
        double mean = std::accumulate(visit_counts.begin(), visit_counts.end(), 0.0) / visit_counts.size();
        double variance = 0.0;
        for (int count : visit_counts) {
            variance += (count - mean) * (count - mean);
        }
        variance /= visit_counts.size();
        
        Logger::Log(LogLevel::DEBUG, "Exploration constant " + std::to_string(c) + 
                   ": Mean visits=" + std::to_string(mean) + 
                   ", Variance=" + std::to_string(variance));
        
        if (c > 1.5 && variance > mean * mean) {
            Logger::Log(LogLevel::ERROR, "High exploration constant not producing uniform enough distribution");
            return;
        }
    }
    
    Logger::Log(LogLevel::TEST, "Exploration test passed");
}

void TestMCTSGameplay() {
    Logger::Log(LogLevel::TEST, "Starting MCTS gameplay test");
    auto state_factory = []() { return std::make_unique<TicTacToeState>(); };
    auto state = std::make_unique<TicTacToeState>();
    std::vector<int> moves;
    
    // Play a complete game
    while (!state->IsTerminal()) {
        // Create new MCTS instance for current state
        auto current_factory = [&state]() { 
            auto new_state = std::make_unique<TicTacToeState>();
            *new_state = *state;
            return new_state;
        };
        MCTS mcts(100, current_factory);
        
        mcts.Search();
        int action = mcts.GetBestAction();
        moves.push_back(action);
        
        Logger::Log(LogLevel::DEBUG, "Selected action: " + std::to_string(action));
        state->Print();
        
        if (action < 0 || action >= 9) {
            Logger::Log(LogLevel::ERROR, "Invalid action selected: " + std::to_string(action));
            return;
        }
        
        state->ApplyAction(action);
    }
    
    // Verify game completion
    if (moves.empty() || moves.size() > 9) {
        Logger::Log(LogLevel::ERROR, "Invalid number of moves: " + std::to_string(moves.size()));
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Gameplay test passed with " + std::to_string(moves.size()) + " moves");
}

int main() {
    TestMCTSTreeConstruction();
    TestMCTSSearchBehavior();
    TestMCTSExploration();
    TestMCTSGameplay();
    return 0;
} 