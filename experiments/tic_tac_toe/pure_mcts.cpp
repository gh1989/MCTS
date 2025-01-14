#include "agents/pure_mcts_agent.h"
#include "agents/random_agent.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "arena/arena_manager.h"
#include "common/logger.h"

int main() {
    Logger::Log(LogLevel::INFO, "=== Testing Pure MCTS Agent ===\n");
    
    // Create initial state and agents
    auto initial_state = std::make_shared<TicTacToeState>();
    auto pure_mcts = std::make_shared<PureMCTSAgent>(10000);  // 10k simulations
    auto random_agent = std::make_shared<RandomAgent>();
    auto arena = ArenaManager(initial_state);
    
    // Test 1: MCTS vs Random
    Logger::Log(LogLevel::INFO, "Testing MCTS (Player 1) vs Random (Player 2)");
    int num_games = 10;
    int mcts_wins = 0, random_wins = 0, draws = 0;
    
    for (int i = 0; i < num_games; ++i) {
        auto result = arena.PlayGame(pure_mcts, random_agent, initial_state, false);
        if (result.winner == 1) mcts_wins++;
        else if (result.winner == -1) random_wins++;
        else draws++;
    }
    
    Logger::Log(LogLevel::INFO, "Results after " + std::to_string(num_games) + " games:");
    Logger::Log(LogLevel::INFO, "MCTS wins: " + std::to_string(mcts_wins));
    Logger::Log(LogLevel::INFO, "Random wins: " + std::to_string(random_wins));
    Logger::Log(LogLevel::INFO, "Draws: " + std::to_string(draws) + "\n");
    
    // Test 2: MCTS vs MCTS
    Logger::Log(LogLevel::INFO, "Testing MCTS vs MCTS (should all be draws)");
    auto pure_mcts2 = std::make_shared<PureMCTSAgent>(10000);  // Same number of simulations

    // Set different seeds for the two agents to ensure different random sequences
    pure_mcts->SetSeed(42);
    pure_mcts2->SetSeed(123);

    int mcts_vs_mcts_games = 10;
    int p1_wins = 0, p2_wins = 0, mcts_draws = 0;
    
    for (int i = 0; i < mcts_vs_mcts_games; ++i) {
        auto result = arena.PlayGame(pure_mcts, pure_mcts2, initial_state, false);
        if (result.winner == 1) p1_wins++;
        else if (result.winner == -1) p2_wins++;
        else mcts_draws++;
    }
    
    Logger::Log(LogLevel::INFO, "Results after " + std::to_string(mcts_vs_mcts_games) + " games:");
    Logger::Log(LogLevel::INFO, "Player 1 wins: " + std::to_string(p1_wins));
    Logger::Log(LogLevel::INFO, "Player 2 wins: " + std::to_string(p2_wins));
    Logger::Log(LogLevel::INFO, "Draws: " + std::to_string(mcts_draws));
    
    return 0;
}
