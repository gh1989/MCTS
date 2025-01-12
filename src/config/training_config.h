#ifndef TRAINING_CONFIG_H_
#define TRAINING_CONFIG_H_

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
    
    // TD Learning parameters
    double td_lambda = 0.7;
    double discount_factor = 0.99;
    
    // File paths
    std::string checkpoint_dir = "checkpoints/";
    std::string log_dir = "logs/";
};

#endif  // TRAINING_CONFIG_H_ 