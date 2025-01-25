#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include "config/training_config.h"
#include <string>
#include <unordered_map>

class CommandLine {
public:
    static void ParseArgs(int argc, char* argv[], TrainingConfig& config) {
        std::unordered_map<std::string, std::string> args;
        
        // Log received arguments
        Logger::Log(LogLevel::INFO, "Received arguments:");
        for (int i = 1; i < argc; i++) {
            Logger::Log(LogLevel::INFO, std::string(argv[i]));
        }
        
        // Parse into map (looking for key=value format)
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            size_t equals_pos = arg.find('=');
            if (equals_pos != std::string::npos) {
                std::string key = arg.substr(0, equals_pos);
                std::string value = arg.substr(equals_pos + 1);
                args[key] = value;
                Logger::Log(LogLevel::INFO, "Parsed " + key + " = " + value);
            }
        }
        
        // Store original values for logging
        auto original = config;
        
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
        
        // Log changes
        LogChanges(original, config);
    }

private:
    static void TryParseInt(const std::unordered_map<std::string, std::string>& args, 
                           const std::string& key, int& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                int new_value = std::stoi(it->second);
                Logger::Log(LogLevel::INFO, "Updating " + key + " from " + 
                          std::to_string(value) + " to " + std::to_string(new_value));
                value = new_value;
            } catch (const std::exception& e) {
                Logger::Log(LogLevel::ERROR, "Failed to parse " + key + ": " + e.what());
            }
        }
    }
    
    static void TryParseDouble(const std::unordered_map<std::string, std::string>& args,
                              const std::string& key, double& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            try {
                double new_value = std::stod(it->second);
                Logger::Log(LogLevel::INFO, "Updating " + key + " from " + 
                          std::to_string(value) + " to " + std::to_string(new_value));
                value = new_value;
            } catch (const std::exception& e) {
                Logger::Log(LogLevel::ERROR, "Failed to parse " + key + ": " + e.what());
            }
        }
    }
    
    static void TryParseString(const std::unordered_map<std::string, std::string>& args,
                              const std::string& key, std::string& value) {
        auto it = args.find(key);
        if (it != args.end()) {
            Logger::Log(LogLevel::INFO, "Updating " + key + " from '" + 
                       value + "' to '" + it->second + "'");
            value = it->second;
        }
    }
    
    static void LogChanges(const TrainingConfig& original, const TrainingConfig& updated) {
        Logger::Log(LogLevel::INFO, "Configuration changes:");
        if (original.num_self_play_games != updated.num_self_play_games) {
            Logger::Log(LogLevel::INFO, "self_play_games: " + 
                       std::to_string(original.num_self_play_games) + " -> " + 
                       std::to_string(updated.num_self_play_games));
        }
        // Add similar checks for other parameters
    }
};

#endif  // COMMAND_LINE_H_ 