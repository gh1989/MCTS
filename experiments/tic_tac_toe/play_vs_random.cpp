#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include "agents/mcts_agent.h"
#include "agents/random_agent.h"
#include "common/command_line.h"
#include "common/logger.h"
#include <iostream>
#include <fstream>
#include <filesystem>

int main(int argc, char* argv[]) {
    // Parse configuration
    TrainingConfig config;
    CommandLine::ParseArgs(argc, argv, config);
    
    Logger::Log(LogLevel::INFO, "Initial checkpoint_dir: " + config.checkpoint_dir);
    
    // Set default model path if not provided or if only directory is provided
    if (config.checkpoint_dir.empty() || config.checkpoint_dir == "checkpoints/") {
        config.checkpoint_dir = "checkpoints/final_network.pt";
        Logger::Log(LogLevel::INFO, "Using default checkpoint path: " + config.checkpoint_dir);
    }
    
    // Ensure path ends with .pt
    if (!config.checkpoint_dir.ends_with(".pt")) {
        Logger::Log(LogLevel::ERROR, "Invalid checkpoint path. Must end with .pt: " + config.checkpoint_dir);
        return 1;
    }
    
    // Add directory existence check before file check
    std::filesystem::path checkpoint_path(config.checkpoint_dir);
    Logger::Log(LogLevel::INFO, "Absolute path: " + std::filesystem::absolute(checkpoint_path).string());
    
    if (!std::filesystem::exists(checkpoint_path.parent_path())) {
        Logger::Log(LogLevel::ERROR, "Checkpoint directory does not exist: " + checkpoint_path.parent_path().string());
        return 1;
    }
    
    // Add file existence check
    if (!std::filesystem::exists(checkpoint_path)) {
        Logger::Log(LogLevel::ERROR, "Model file not found at: " + checkpoint_path.string());
        return 1;
    }
    
    // Initialize game and network
    auto state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    // Determine device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        Logger::Log(LogLevel::INFO, "Using CUDA device");
    } else {
        Logger::Log(LogLevel::INFO, "Using CPU device");
    }
    
    network->to(device);  // Move network to appropriate device
    
    // Load the trained model
    std::string model_path = config.checkpoint_dir;
    
    try {
        Logger::Log(LogLevel::INFO, "Attempting to load model from: " + model_path);
        torch::load(network, model_path);
        network->to(device);  // Ensure model is on correct device after loading
        Logger::Log(LogLevel::INFO, "Loaded final model from: " + model_path);
        
        // Test network
        auto test_tensor = state->ToTensor().to(device);  // Move input tensor to same device
        auto [policy, value] = network->forward(test_tensor);
        Logger::Log(LogLevel::INFO, "Network test - Initial position value: " + 
                   std::to_string(value.item<float>()));
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Failed to load model: " + std::string(e.what()));
        return 1;
    }
    
    // Create agents with more simulations
    config.simulations_per_move = 10000;
    Logger::Log(LogLevel::INFO, "Using " + std::to_string(config.simulations_per_move) + 
                " simulations per move");
    auto mcts_agent = std::make_shared<MCTSAgent>(network, config);
    auto random_agent = std::make_shared<RandomAgent>();
    
    // Game statistics
    int num_games = 12;
    int mcts_wins = 0;
    int random_wins = 0;
    int draws = 0;
    
    // Play games
    for (int game = 0; game < num_games; game++) {
        state = std::make_shared<TicTacToeState>();
        bool mcts_plays_first = (game % 2 == 0);  // Alternate who goes first
        
        Logger::Log(LogLevel::INFO, "\n=== Game " + std::to_string(game + 1) + " ===");
        Logger::Log(LogLevel::INFO, (mcts_plays_first ? "MCTS plays first" : "Random plays first"));
        
        while (!state->IsTerminal()) {
            state->Print();
            
            int action;
            if ((state->GetCurrentPlayer() == 1) == mcts_plays_first) {
                Logger::Log(LogLevel::INFO, "MCTS agent thinking...");
                action = mcts_agent->GetAction(state);
                Logger::Log(LogLevel::INFO, "MCTS plays: " + std::to_string(action));
            } else {
                action = random_agent->GetAction(state);
                Logger::Log(LogLevel::INFO, "Random plays: " + std::to_string(action));
            }
            
            state->ApplyAction(action);
        }
        
        // Game over
        state->Print();
        double outcome = state->Evaluate();
        
        if (outcome == 0) {
            Logger::Log(LogLevel::INFO, "Game is a draw!");
            draws++;
        } else if ((outcome > 0) == mcts_plays_first) {
            Logger::Log(LogLevel::INFO, "MCTS wins!");
            mcts_wins++;
        } else {
            Logger::Log(LogLevel::INFO, "Random wins!");
            random_wins++;
        }
        
        // Print current statistics
        Logger::Log(LogLevel::INFO, "\nCurrent stats after " + std::to_string(game + 1) + " games:");
        Logger::Log(LogLevel::INFO, "MCTS wins: " + std::to_string(mcts_wins) + 
                   " (" + std::to_string(100.0 * mcts_wins / (game + 1)) + "%)");
        Logger::Log(LogLevel::INFO, "Random wins: " + std::to_string(random_wins) + 
                   " (" + std::to_string(100.0 * random_wins / (game + 1)) + "%)");
        Logger::Log(LogLevel::INFO, "Draws: " + std::to_string(draws) + 
                   " (" + std::to_string(100.0 * draws / (game + 1)) + "%)");
    }
    
    // Print final statistics
    Logger::Log(LogLevel::INFO, "\n=== Final Results ===");
    Logger::Log(LogLevel::INFO, "Total games: " + std::to_string(num_games));
    Logger::Log(LogLevel::INFO, "MCTS wins: " + std::to_string(mcts_wins) + 
               " (" + std::to_string(100.0 * mcts_wins / num_games) + "%)");
    Logger::Log(LogLevel::INFO, "Random wins: " + std::to_string(random_wins) + 
               " (" + std::to_string(100.0 * random_wins / num_games) + "%)");
    Logger::Log(LogLevel::INFO, "Draws: " + std::to_string(draws) + 
               " (" + std::to_string(100.0 * draws / num_games) + "%)");
    
    return 0;
} 