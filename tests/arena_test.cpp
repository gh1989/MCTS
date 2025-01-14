#include "arena/arena_manager.h"
#include "agents/agent_factory.h"
#include "common/logger.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>

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
    auto result = arena.PlayGame(agent1, agent2, initial_state, true);
    
    // Verify history properties
    if (result.game_history.empty()) {
        Logger::Log(LogLevel::ERROR, "Game history is empty");
        return;
    }
    
    // Check state progression
    int move_num = 0;
    for (const auto& [state, outcome] : result.game_history) {
        Logger::Log(LogLevel::DEBUG, "Move " + std::to_string(move_num) + 
                   ", Outcome: " + std::to_string(outcome));
        state->Print();
        move_num++;
    }
    
    Logger::Log(LogLevel::TEST, "Arena game history test passed");
}

void TestArenaDifferentAgents() {
    Logger::Log(LogLevel::TEST, "Starting arena different agents test");
    
    TrainingConfig config;
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    auto network = std::make_shared<TicTacToeNetwork>();
    
    // Test different agent combinations
    std::vector<std::pair<std::string, std::string>> agent_pairs = {
        {"random", "random"},
        {"mcts", "random"},
        {"mcts", "mcts"}
    };
    
    for (const auto& [agent1_type, agent2_type] : agent_pairs) {
        Logger::Log(LogLevel::DEBUG, "Testing " + agent1_type + " vs " + agent2_type);
        
        auto agent1 = agent1_type == "mcts" ? 
            AgentFactory::CreateAgent(agent1_type, config, network) :
            AgentFactory::CreateAgent(agent1_type, config);
            
        auto agent2 = agent2_type == "mcts" ? 
            AgentFactory::CreateAgent(agent2_type, config, network) :
            AgentFactory::CreateAgent(agent2_type, config);
        
        auto result = arena.PlayGame(agent1, agent2, initial_state, true);
        
        Logger::Log(LogLevel::DEBUG, agent1_type + " vs " + agent2_type + 
                   " winner: " + std::to_string(result.winner));
    }
    
    Logger::Log(LogLevel::TEST, "Arena different agents test passed");
}

void TestArenaGameValidation() {
    Logger::Log(LogLevel::TEST, "Starting arena game validation test");
    
    TrainingConfig config;
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create agents
    auto agent1 = AgentFactory::CreateAgent("random", config);
    auto agent2 = AgentFactory::CreateAgent("random", config);
    
    // Play multiple games and validate results
    int num_games = 10;
    std::vector<int> winners;
    std::vector<size_t> game_lengths;
    
    for (int i = 0; i < num_games; i++) {
        auto result = arena.PlayGame(agent1, agent2, initial_state, true);
        
        // Validate winner
        if (result.winner != 1 && result.winner != -1 && result.winner != 0) {
            Logger::Log(LogLevel::ERROR, "Invalid winner: " + std::to_string(result.winner));
            return;
        }
        
        // Validate game length
        if (result.game_history.size() < 5 || result.game_history.size() > 10) {
            Logger::Log(LogLevel::ERROR, "Invalid game length: " + 
                       std::to_string(result.game_history.size()));
            return;
        }
        
        winners.push_back(result.winner);
        game_lengths.push_back(result.game_history.size());
    }
    
    // Check for variety in outcomes
    std::sort(winners.begin(), winners.end());
    auto unique_winners = std::unique(winners.begin(), winners.end());
    int unique_outcomes = std::distance(winners.begin(), unique_winners);
    
    if (unique_outcomes < 2) {
        Logger::Log(LogLevel::ERROR, "Not enough variety in game outcomes");
        return;
    }
    
    Logger::Log(LogLevel::DEBUG, "Average game length: " + 
               std::to_string(std::accumulate(game_lengths.begin(), 
                                            game_lengths.end(), 0.0) / num_games));
    
    Logger::Log(LogLevel::TEST, "Arena game validation test passed");
}

void TestPlayerAlternation() {
    Logger::Log(LogLevel::TEST, "Starting player alternation test");
    
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create two agents
    auto agent1 = std::make_shared<RandomAgent>();
    auto agent2 = std::make_shared<RandomAgent>();
    
    // Play a game and verify moves directly
    auto state = std::make_shared<TicTacToeState>();
    
    // Track moves as they happen
    while (!state->IsTerminal()) {
        int current_player = state->GetCurrentPlayer();
        Logger::Log(LogLevel::DEBUG, "Before move - Current player: " + std::to_string(current_player));
        
        auto valid_actions = state->GetValidActions();
        if (valid_actions.empty()) break;
        
        int action = valid_actions[0];  // Just take first valid action for testing
        state->ApplyAction(action);
        
        int next_player = state->GetCurrentPlayer();
        Logger::Log(LogLevel::DEBUG, "After move - Current player: " + std::to_string(next_player));
        
        if (next_player == current_player) {
            Logger::Log(LogLevel::ERROR, "Player not alternating after move");
            Logger::Log(LogLevel::ERROR, "Before move: " + std::to_string(current_player) + 
                       ", After move: " + std::to_string(next_player));
            return;
        }
    }
    
    Logger::Log(LogLevel::TEST, "Player alternation test passed");
}

void TestGameHistoryAndAlternation() {
    Logger::Log(LogLevel::TEST, "Starting game history and alternation test");
    
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create two agents
    auto agent1 = std::make_shared<RandomAgent>();
    auto agent2 = std::make_shared<RandomAgent>();
    
    // Play game with history recording
    auto result = arena.PlayGame(agent1, agent2, initial_state, true);
    
    // Verify history properties
    if (result.game_history.empty()) {
        Logger::Log(LogLevel::ERROR, "Game history is empty");
        return;
    }
    
    // Check player alternation in history
    int expected_player = 1;  // First player should be 1
    Logger::Log(LogLevel::DEBUG, "Checking player alternation in history:");
    
    for (size_t i = 0; i < result.game_history.size(); ++i) {
        const auto& [state, outcome] = result.game_history[i];
        int current_player = state->GetCurrentPlayer();
        
        Logger::Log(LogLevel::DEBUG, 
            "Move " + std::to_string(i) + 
            ", Expected player: " + std::to_string(expected_player) + 
            ", Actual player: " + std::to_string(current_player));
        
        state->Print();
        
        if (current_player != expected_player) {
            Logger::Log(LogLevel::ERROR, 
                "Player alternation error at move " + std::to_string(i) + 
                ". Expected: " + std::to_string(expected_player) + 
                ", Got: " + std::to_string(current_player));
            return;
        }
        
        expected_player = -expected_player;  // Switch expected player
    }
    
    // Verify game length
    if (result.game_history.size() > 10) {  // Initial state + max 9 moves
        Logger::Log(LogLevel::ERROR, 
            "Game history too long: " + std::to_string(result.game_history.size()));
        return;
    }
    
    // Verify final state is terminal
    const auto& final_state = result.game_history.back().first;
    if (!final_state->IsTerminal()) {
        Logger::Log(LogLevel::ERROR, "Final state is not terminal");
        return;
    }
    
    Logger::Log(LogLevel::TEST, 
        "Game history and alternation test passed with " + 
        std::to_string(result.game_history.size()) + " states");
}

void TestSingleGame() {
    Logger::Log(LogLevel::TEST, "Starting single game test");
    
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create two agents
    TrainingConfig config;
    auto network = std::make_shared<TicTacToeNetwork>();
    auto player1 = AgentFactory::CreateAgent("mcts", config, network);
    auto player2 = AgentFactory::CreateAgent("random", config);
    
    // Play a single game
    auto result = arena.PlayGame(player1, player2, initial_state, false);
    
    // Verify result properties
    if (result.winner != 1 && result.winner != -1 && result.winner != 0) {
        Logger::Log(LogLevel::ERROR, "Invalid game result");
        return;
    }
    
    if (!result.player1 || !result.player2) {
        Logger::Log(LogLevel::ERROR, "Missing player references in result");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Single game test passed");
}

void TestEloRating() {
    Logger::Log(LogLevel::TEST, "Starting Elo rating test");
    
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create agents
    TrainingConfig config;
    auto network = std::make_shared<TicTacToeNetwork>();
    auto player1 = AgentFactory::CreateAgent("mcts", config, network);
    auto player2 = AgentFactory::CreateAgent("random", config);
    
    // Get initial ratings
    double initial_rating1 = arena.GetAgentRating(player1);
    double initial_rating2 = arena.GetAgentRating(player2);
    
    // Play a game and update ratings
    auto result = arena.PlayGame(player1, player2, initial_state, false);
    arena.UpdateRatings(result);
    
    // Get updated ratings
    double new_rating1 = arena.GetAgentRating(player1);
    double new_rating2 = arena.GetAgentRating(player2);
    
    // Verify rating changes
    if (result.winner == 1) {
        if (new_rating1 <= initial_rating1 || new_rating2 >= initial_rating2) {
            Logger::Log(LogLevel::ERROR, "Invalid rating update for win");
            return;
        }
    } else if (result.winner == -1) {
        if (new_rating1 >= initial_rating1 || new_rating2 <= initial_rating2) {
            Logger::Log(LogLevel::ERROR, "Invalid rating update for loss");
            return;
        }
    }
    
    Logger::Log(LogLevel::TEST, "Elo rating test passed");
}

void TestSmallTournament() {
    Logger::Log(LogLevel::TEST, "Starting small tournament test");
    
    auto initial_state = std::make_shared<TicTacToeState>();
    ArenaManager arena(initial_state);
    
    // Create multiple agents
    TrainingConfig config;
    std::vector<std::shared_ptr<Agent>> agents;
    auto network = std::make_shared<TicTacToeNetwork>();
    agents.push_back(AgentFactory::CreateAgent("mcts", config, network));
    agents.push_back(AgentFactory::CreateAgent("random", config));
    agents.push_back(AgentFactory::CreateAgent("random", config));
    
    // Run small tournament
    int games_per_matchup = 2;
    auto results = arena.RunTournament(agents, initial_state, games_per_matchup);
    
    // Verify results
    int expected_games = (agents.size() * (agents.size() - 1)) * games_per_matchup;
    if (results.size() != expected_games) {
        Logger::Log(LogLevel::ERROR, "Incorrect number of tournament games played");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Small tournament test passed");
}

int main() {
    TestSingleGame();
    TestEloRating();
    TestSmallTournament();
    TestArenaGameHistory();
    return 0;
} 