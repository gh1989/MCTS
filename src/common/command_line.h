#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include "config/training_config.h"
#include <string>
#include <unordered_map>

class CommandLine {
public:
    static void ParseArgs(int argc, char* argv[], TrainingConfig& config) {
        std::unordered_map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 < argc) {
                args[argv[i]] = argv[i + 1];
            }
        }
        
        // Network params
        TryParseInt(args, "--num-filters", config.num_filters);
        TryParseInt(args, "--num-residual-blocks", config.num_residual_blocks);
        TryParseDouble(args, "--learning-rate", config.learning_rate);
        
        // MCTS params
        TryParseInt(args, "--simulations", config.simulations_per_move);
        TryParseDouble(args, "--exploration", config.exploration_constant);
        TryParseDouble(args, "--temperature", config.temperature);
        
        // Training params
        TryParseInt(args, "--self-play-games", config.num_self_play_games);
        TryParseInt(args, "--eval-games", config.games_per_evaluation);
        TryParseDouble(args, "--required-win-rate", config.required_win_rate);
        TryParseInt(args, "--total-iterations", config.total_iterations);
        TryParseInt(args, "--training-steps", config.training_steps);
        
        // Paths
        TryParseString(args, "--checkpoint-dir", config.checkpoint_dir);
        TryParseString(args, "--log-dir", config.log_dir);
        
        // Batch size
        TryParseInt(args, "--batch-size", config.batch_size);
    }

private:
    static void TryParseInt(const std::unordered_map<std::string, std::string>& args, 
                           const std::string& key, int& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            value = std::stoi(it->second);
        }
    }
    
    static void TryParseDouble(const std::unordered_map<std::string, std::string>& args,
                              const std::string& key, double& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            value = std::stod(it->second);
        }
    }
    
    static void TryParseString(const std::unordered_map<std::string, std::string>& args,
                              const std::string& key, std::string& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            value = it->second;
        }
    }
};

#endif  // COMMAND_LINE_H_ 