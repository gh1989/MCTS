#include "training/training_manager.h"
#include "agents/agent_factory.h"
#include "common/logger.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>

void TestSingleGameIteration() {
    Logger::Log(LogLevel::TEST, "Starting single game iteration test");
    
    TrainingConfig config;
    config.num_self_play_games = 1;  // Just test one game
    config.simulations_per_move = 10;  // Keep it fast for testing
    
    std::shared_ptr<TicTacToeState> initial_state = std::make_shared<TicTacToeState>();
    std::shared_ptr<TicTacToeNetwork> network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    // Create agents directly to test their behavior
    std::shared_ptr<MCTSAgent> agent = std::make_shared<MCTSAgent>(network, config);
    agent->SetTrainingMode(true);
    
    // Play a single game manually to verify moves
    std::shared_ptr<State> state = std::shared_ptr<State>(initial_state->Clone());
    while (!state->IsTerminal()) {
        auto valid_actions = state->GetValidActions();
        Logger::Log(LogLevel::DEBUG, "Valid actions before move:");
        for (int action : valid_actions) {
            Logger::Log(LogLevel::DEBUG, std::to_string(action) + " ");
        }
        
        int action = agent->GetAction(state);
        Logger::Log(LogLevel::DEBUG, "Agent selected action: " + std::to_string(action));
        
        if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end()) {
            Logger::Log(LogLevel::ERROR, "Agent selected invalid action: " + std::to_string(action));
            state->Print();
            return;
        }
        
        state->ApplyAction(action);
        state->Print();
    }
    
    Logger::Log(LogLevel::TEST, "Single game iteration test passed");
}

void TestTrainingIteration() {
    Logger::Log(LogLevel::TEST, "Starting training iteration test");
    
    TrainingConfig config;
    config.num_self_play_games = 10;  // Reduced for testing
    config.games_per_evaluation = 5;
    config.training_steps = 32;
    config.log_frequency = 1;
    config.checkpoint_dir = "test_checkpoints/";
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    TrainingManager manager(config, initial_state, network);
    
    try {
        manager.RunTrainingIteration();
        Logger::Log(LogLevel::TEST, "Training iteration test passed");
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Training iteration failed: " + std::string(e.what()));
        return;
    }
}

void TestShortTrainingIteration() {
    Logger::Log(LogLevel::TEST, "Starting short training iteration test");
    
    TrainingConfig config;
    config.num_self_play_games = 3;  // Minimum games for testing
    config.games_per_evaluation = 2;
    config.training_steps = 16;      // Reduced steps
    config.log_frequency = 1;
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    TrainingManager manager(config, initial_state, network);
    
    try {
        manager.RunTrainingIteration();
        Logger::Log(LogLevel::TEST, "Short training iteration test passed");
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Short training iteration failed: " + std::string(e.what()));
        return;
    }
}

void TestLongTrainingIteration() {
    Logger::Log(LogLevel::TEST, "Starting long training iteration test");
    
    TrainingConfig config;
    config.num_self_play_games = 20;  // More games for thorough testing
    config.games_per_evaluation = 10;
    config.training_steps = 64;       // More training steps
    config.log_frequency = 1;
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    TrainingManager manager(config, initial_state, network);
    
    try {
        manager.RunTrainingIteration();
        Logger::Log(LogLevel::TEST, "Long training iteration test passed");
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Long training iteration failed: " + std::string(e.what()));
        return;
    }
}

void TestHighExplorationTraining() {
    Logger::Log(LogLevel::TEST, "Starting high exploration training test");
    
    TrainingConfig config;
    config.num_self_play_games = 10;
    config.games_per_evaluation = 5;
    config.training_steps = 32;
    config.exploration_constant = 2.0;  // Higher exploration
    config.log_frequency = 1;
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    TrainingManager manager(config, initial_state, network);
    
    try {
        manager.RunTrainingIteration();
        Logger::Log(LogLevel::TEST, "High exploration training test passed");
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "High exploration training failed: " + std::string(e.what()));
        return;
    }
}

void TestMCTSExplorationDuringTraining() {
    Logger::Log(LogLevel::TEST, "Starting MCTS exploration test");
    
    TrainingConfig config;
    config.simulations_per_move = 800;
    config.exploration_constant = 2.0;
    config.temperature = 1.5;
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCPU);
    
    auto agent = std::make_shared<MCTSAgent>(network, config);
    agent->SetTrainingMode(true);
    
    // Track unique actions taken from the initial position
    std::vector<int> initial_actions;
    const int num_test_games = 10;
    
    for (int game = 0; game < num_test_games; ++game) {
        auto state = std::shared_ptr<State>(initial_state->Clone());
        int first_action = agent->GetAction(state);
        initial_actions.push_back(first_action);
        
        // Play out the rest of the game to ensure proper cleanup
        while (!state->IsTerminal()) {
            auto valid_actions = state->GetValidActions();
            int action = agent->GetAction(state);
            
            if (std::find(valid_actions.begin(), valid_actions.end(), action) 
                == valid_actions.end()) {
                Logger::Log(LogLevel::ERROR, "Invalid action selected: " + 
                          std::to_string(action));
                return;
            }
            
            state->ApplyAction(action);
            state->Print();
        }
    }
    
    // Check exploration by counting unique first moves
    std::sort(initial_actions.begin(), initial_actions.end());
    auto unique_end = std::unique(initial_actions.begin(), initial_actions.end());
    int unique_count = std::distance(initial_actions.begin(), unique_end);
    
    Logger::Log(LogLevel::TEST, "Unique first moves: " + 
                std::to_string(unique_count) + "/" + 
                std::to_string(num_test_games));
    
    // Test fails if we don't see at least 3 different first moves
    if (unique_count < 3) {
        Logger::Log(LogLevel::ERROR, "MCTS exploration during training is insufficient");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "MCTS exploration test passed");
}

int main() {
    TestSingleGameIteration();
    TestTrainingIteration();
    TestShortTrainingIteration();
    TestLongTrainingIteration();
    TestHighExplorationTraining();
    TestMCTSExplorationDuringTraining();
    return 0;
} 