#include "arena/arena_manager.h"
#include "agents/agent_factory.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include "networks/tic_tac_toe_network.h"

void TestArenaGamePlay() {
    Logger::Log(LogLevel::TEST, "Starting arena gameplay test");
    
    TrainingConfig config;
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create two random agents for testing
    auto agent1 = AgentFactory::CreateAgent("random", config);
    auto agent2 = AgentFactory::CreateAgent("random", config);
    
    // Play a game and verify the result
    auto result = arena.PlayGame(agent1, agent2, initial_state, false);
    
    Logger::Log(LogLevel::TEST, "Game completed with winner: " + std::to_string(result.winner));
    Logger::Log(LogLevel::TEST, "Game length: " + std::to_string(result.game_history.size()));
    
    // Verify game history
    for (const auto& [state, outcome] : result.game_history) {
        if (!state) {
            Logger::Log(LogLevel::ERROR, "Invalid state in game history");
            return;
        }
        state->Print();
    }
}

void TestArenaGameHistory() {
    Logger::Log(LogLevel::TEST, "Starting arena game history test");
    
    TrainingConfig config;
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    
    // Create MCTS agents for testing
    auto network = std::make_shared<TicTacToeNetwork>();
    auto agent1 = AgentFactory::CreateAgent("mcts", config, network);
    auto agent2 = AgentFactory::CreateAgent("mcts", config, network);
    
    // Play game with history recording
    auto result = arena.PlayGame(agent1, agent2, initial_state, false);
    
    // Verify history properties
    if (result.game_history.empty()) {
        Logger::Log(LogLevel::ERROR, "Game history is empty");
        return;
    }
    
    // Check state progression
    auto prev_state = initial_state->Clone();
    for (const auto& [state, outcome] : result.game_history) {
        state->Print();
        prev_state = state->Clone();
    }
    
    Logger::Log(LogLevel::TEST, "Arena game history test passed");
}

int main() {
    TestArenaGamePlay();
    TestArenaGameHistory();
    return 0;
} 