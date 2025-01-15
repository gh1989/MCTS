#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include "training/training_manager.h"
#include "common/command_line.h"
#include "common/logger.h"
#include "common/gpu.h"
#include <filesystem>

int main(int argc, char* argv[]) {
    // 1. Parse and log configuration
    TrainingConfig config;
    CommandLine::ParseArgs(argc, argv, config);
    
    Logger::Log(LogLevel::INFO, "=== Starting Tic-Tac-Toe Training ===");
    config.LogConfig();
    
    // Log initial GPU status
    Logger::Log(LogLevel::INFO, "=== GPU Information ===");
    LogGPUStats();

    // 2. Initialize game state and network
    Logger::Log(LogLevel::INFO, "Initializing game state and network...");
    auto initial_state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    // 3. Create training manager
    Logger::Log(LogLevel::INFO, "Setting up training manager...");
    TrainingManager trainer(config, initial_state, network);
    
    // 4. Load existing checkpoint if available
    std::string checkpoint_path = config.checkpoint_dir + "/final_network.pt";
    if (std::filesystem::exists(checkpoint_path)) {
        Logger::Log(LogLevel::INFO, "Loading existing checkpoint from: " + checkpoint_path);
        trainer.LoadCheckpoint(checkpoint_path);
    } else {
        throw std::runtime_error("No existing checkpoint found. Starting from scratch.");
        //Logger::Log(LogLevel::INFO, "No existing checkpoint found. Starting from scratch.");
    }
    
    // 5. Training loop
    Logger::Log(LogLevel::INFO, "Beginning training loop...");
    
    for (int iteration = 0; iteration < config.total_iterations; ++iteration) {
        Logger::Log(LogLevel::INFO, 
            "\n=== Training Iteration " + std::to_string(iteration + 1) + 
            "/" + std::to_string(config.total_iterations) + " ===");
        
        // Each iteration consists of:
        // a. Self-play to generate training data
        // b. Network training on the collected data
        // c. Evaluation against previous best network
        // d. Checkpoint saving if improved
        trainer.RunTrainingIteration();
        
        // Log GPU stats every N iterations
        if (iteration % 10 == 0) {
            Logger::Log(LogLevel::INFO, "=== GPU Status Update ===");
            LogGPUStats();
        }
    }
    
    // Add final model saving
    Logger::Log(LogLevel::INFO, "Training completed! Saving final model...");
    std::string final_model_path = config.checkpoint_dir + "/final_network.pt";
    trainer.SaveCheckpoint(final_model_path);
    Logger::Log(LogLevel::INFO, "Final model saved to: " + final_model_path);
    
    // Log final GPU stats
    Logger::Log(LogLevel::INFO, "=== Final GPU Status ===");
    LogGPUStats();
    
    return 0;
}