#include "training/training_manager.h"
#include "common/logger.h"
#include <filesystem>
#include <algorithm>

TrainingManager::TrainingManager(const TrainingConfig& config)
    : config_(config) {
    // Create checkpoint directory if it doesn't exist
    std::filesystem::create_directories(config.checkpoint_dir);
    
    // Initialize agents with same network architecture
    auto network = std::make_shared<TicTacToeNetwork>();
    best_agent_ = std::make_shared<MCTSAgent>(network);
    training_agent_ = std::make_shared<MCTSAgent>(network->clone());
}

void TrainingManager::RunTrainingIteration() {
    Logger::Log(LogLevel::INFO, "Starting self-play phase");
    self_play_buffer_.clear();
    
    // Generate self-play games
    for (int game = 0; game < config_.num_self_play_games; ++game) {
        auto result = arena_.PlayGame(training_agent_, training_agent_, true);
        self_play_buffer_.insert(
            self_play_buffer_.end(),
            result.game_history.begin(),
            result.game_history.end()
        );
    }
    
    Logger::Log(LogLevel::INFO, "Training network on " + 
        std::to_string(self_play_buffer_.size()) + " positions");
    
    // Train network on collected data
    training_agent_->TrainOnBuffer(self_play_buffer_);
    
    // Evaluate if new network is better
    if (EvaluateNewNetwork()) {
        Logger::Log(LogLevel::INFO, "New network accepted as best");
        SaveCheckpoint(config_.checkpoint_dir + "/best_network.pt");
        best_agent_ = std::make_shared<MCTSAgent>(training_agent_->CloneNetwork());
    } else {
        Logger::Log(LogLevel::INFO, "New network rejected, reverting");
        training_agent_ = std::make_shared<MCTSAgent>(best_agent_->CloneNetwork());
    }
}

bool TrainingManager::EvaluateNewNetwork() {
    Logger::Log(LogLevel::INFO, "Evaluating new network");
    
    int wins = 0, losses = 0, draws = 0;
    
    for (int game = 0; game < config_.games_per_evaluation; ++game) {
        auto result = (game % 2 == 0)
            ? arena_.PlayGame(training_agent_, best_agent_)
            : arena_.PlayGame(best_agent_, training_agent_);
            
        if (game % 2 == 1) result.winner = -result.winner;
        
        if (result.winner == 1) wins++;
        else if (result.winner == -1) losses++;
        else draws++;
    }
    
    double win_rate = (wins + 0.5 * draws) / config_.games_per_evaluation;
    Logger::Log(LogLevel::INFO, "Win rate: " + std::to_string(win_rate));
    
    return win_rate >= config_.required_win_rate;
}

void TrainingManager::SaveCheckpoint(const std::string& filepath) {
    training_agent_->SaveModel(filepath);
}

void TrainingManager::LoadCheckpoint(const std::string& filepath) {
    training_agent_->LoadModel(filepath);
    best_agent_->LoadModel(filepath);
} 