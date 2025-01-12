#ifndef TRAINING_CONFIG_H_
#define TRAINING_CONFIG_H_

#include "common/logger.h"
#include <string>

struct TrainingConfig {
    // Network parameters
    int input_channels = 3;
    int num_filters = 32;
    int num_residual_blocks = 3;
    double learning_rate = 1e-4;
    
    // MCTS parameters
    int simulations_per_move = 800;
    double exploration_constant = 1.4142;  // sqrt(2)
    double temperature = 1.0;
    
    // Training parameters
    int num_self_play_games = 1000;
    int games_per_evaluation = 100;
    double required_win_rate = 0.55;
    int batch_size = 256;
    int training_steps = 1000;
    int log_frequency = 100;
    int total_iterations = 100;
    
    // TD Learning parameters
    double td_lambda = 0.7;
    double discount_factor = 0.99;
    
    // File paths
    std::string checkpoint_dir = "checkpoints/";
    std::string log_dir = "logs/";

    // Add this method to log the configuration
    void LogConfig() const {
        Logger::Log(LogLevel::INFO, "=== Training Configuration ===");
        
        Logger::Log(LogLevel::INFO, "Network Parameters:");
        Logger::Log(LogLevel::INFO, "  - Number of filters: " + std::to_string(num_filters));
        Logger::Log(LogLevel::INFO, "  - Residual blocks: " + std::to_string(num_residual_blocks));
        Logger::Log(LogLevel::INFO, "  - Learning rate: " + std::to_string(learning_rate));
        
        Logger::Log(LogLevel::INFO, "MCTS Parameters:");
        Logger::Log(LogLevel::INFO, "  - Simulations per move: " + std::to_string(simulations_per_move));
        Logger::Log(LogLevel::INFO, "  - Exploration constant: " + std::to_string(exploration_constant));
        Logger::Log(LogLevel::INFO, "  - Temperature: " + std::to_string(temperature));
        
        Logger::Log(LogLevel::INFO, "Training Parameters:");
        Logger::Log(LogLevel::INFO, "  - Self-play games: " + std::to_string(num_self_play_games));
        Logger::Log(LogLevel::INFO, "  - Evaluation games: " + std::to_string(games_per_evaluation));
        Logger::Log(LogLevel::INFO, "  - Required win rate: " + std::to_string(required_win_rate));
        
        Logger::Log(LogLevel::INFO, "Paths:");
        Logger::Log(LogLevel::INFO, "  - Checkpoint directory: " + checkpoint_dir);
        Logger::Log(LogLevel::INFO, "  - Log directory: " + log_dir);
        
        Logger::Log(LogLevel::INFO, "===========================");
    }
};

#endif  // TRAINING_CONFIG_H_ 