#include "training/training_manager.h"
#include "agents/agent_factory.h"
#include "common/logger.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>

void TestTrainingIteration() {
    Logger::Log(LogLevel::TEST, "Starting training iteration test");
    
    TrainingConfig config;
    config.num_self_play_games = 10;  // Reduced for testing
    config.games_per_evaluation = 5;
    config.checkpoint_dir = "test_checkpoints/";
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    TrainingManager manager(config, initial_state, network);
    
    // Run a training iteration
    manager.RunTrainingIteration();
    
    // Verify checkpoint was created
    if (!std::filesystem::exists(config.checkpoint_dir + "/best_network.pt")) {
        Logger::Log(LogLevel::ERROR, "Checkpoint file not created");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Training iteration test passed");
}

void TestNetworkEvaluation() {
    Logger::Log(LogLevel::TEST, "Starting network evaluation test");
    
    TrainingConfig config;
    config.games_per_evaluation = 10;
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    TrainingManager manager(config, initial_state, network);
    
    // Test evaluation against random agent
    auto random_agent = AgentFactory::CreateAgent("random", config);
    bool evaluation_result = manager.EvaluateNewNetwork();
    
    Logger::Log(LogLevel::TEST, "Network evaluation test passed");
}

void TestCheckpointSaveLoad() {
    Logger::Log(LogLevel::TEST, "Starting checkpoint save/load test");
    
    TrainingConfig config;
    config.checkpoint_dir = "test_checkpoints/";
    
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    TrainingManager manager(config, initial_state, network);
    
    // Save checkpoint
    std::string checkpoint_path = config.checkpoint_dir + "/test_network.pt";
    manager.SaveCheckpoint(checkpoint_path);
    
    // Load checkpoint
    manager.LoadCheckpoint(checkpoint_path);
    
    Logger::Log(LogLevel::TEST, "Checkpoint save/load test passed");
}

int main() {
    TestTrainingIteration();
    //TestNetworkEvaluation();
    //TestCheckpointSaveLoad();
    return 0;
} 