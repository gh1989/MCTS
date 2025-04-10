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
    int batch_size = 256;
    int num_self_play_games = 1000;
    int games_per_evaluation = 100;
    double required_win_rate = 0.55;
    int training_steps = 1000;
    int log_frequency = 100;
    int total_iterations = 100;
    
    // TD Learning parameters
    double td_lambda = 0.7;
    double discount_factor = 0.99;
    
    // Replay buffer size
    int replay_buffer_size = 10000;

    // File paths
    std::string checkpoint_dir = "checkpoints/";
    std::string log_dir = "logs/";

    // Learning rate parameters
    double l2_reg_weight = 0.0001;
    double lr_decay_rate = 0.95;
    int lr_decay_steps = 1000;

    // Add this method to log the configuration
    void LogConfig() const {
        Logger::Log(LogLevel::INFO, "\n=== Training Configuration ===");
        
        // Training parameters
        Logger::Log(LogLevel::INFO, "Training Parameters:");
        Logger::Log(LogLevel::INFO, "  Total Iterations: " + std::to_string(total_iterations));
        Logger::Log(LogLevel::INFO, "  Self-play Games per Iteration: " + std::to_string(num_self_play_games));
        Logger::Log(LogLevel::INFO, "  Training Steps per Iteration: " + std::to_string(training_steps));
        Logger::Log(LogLevel::INFO, "  Batch Size: " + std::to_string(batch_size));
        Logger::Log(LogLevel::INFO, "  Evaluation Games: " + std::to_string(games_per_evaluation));
        Logger::Log(LogLevel::INFO, "  Required Win Rate: " + std::to_string(required_win_rate));
        
        // MCTS parameters
        Logger::Log(LogLevel::INFO, "\nMCTS Parameters:");
        Logger::Log(LogLevel::INFO, "  Simulations per Move: " + std::to_string(simulations_per_move));
        Logger::Log(LogLevel::INFO, "  Exploration Constant: " + std::to_string(exploration_constant));
        Logger::Log(LogLevel::INFO, "  Temperature: " + std::to_string(temperature));
        
        // Network parameters
        Logger::Log(LogLevel::INFO, "\nNetwork Parameters:");
        Logger::Log(LogLevel::INFO, "  Learning Rate: " + std::to_string(learning_rate));
        Logger::Log(LogLevel::INFO, "  Number of Filters: " + std::to_string(num_filters));
        Logger::Log(LogLevel::INFO, "  Residual Blocks: " + std::to_string(num_residual_blocks));
        
        // Paths
        Logger::Log(LogLevel::INFO, "\nPaths:");
        Logger::Log(LogLevel::INFO, "  Checkpoint Directory: " + checkpoint_dir);
        Logger::Log(LogLevel::INFO, "  Log Directory: " + log_dir);
        
        Logger::Log(LogLevel::INFO, "===========================\n");
    }
};

#endif  // TRAINING_CONFIG_H_ 