#include "training/training_manager.h"
#include "common/logger.h"
#include "common/network.h"
#include "agents/mcts_agent.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>
#include <algorithm>
#include <chrono>

TrainingManager::TrainingManager(const TrainingConfig& config,
                               std::shared_ptr<State> initial_state,
                               std::shared_ptr<ValuePolicyNetwork> network)
    : config_(config),
      initial_state_(initial_state),
      arena_(initial_state) {
    std::filesystem::create_directories(config.checkpoint_dir);
    
    training_agent_ = std::make_shared<MCTSAgent>(
        std::dynamic_pointer_cast<ValuePolicyNetwork>(network), 
        config);
    best_agent_ = std::make_shared<MCTSAgent>(
        std::dynamic_pointer_cast<ValuePolicyNetwork>(network->clone()), 
        config);
    
    training_agent_->SetTrainingMode(true);
    best_agent_->SetTrainingMode(false);
}

void TrainingManager::RunTrainingIteration() {
    training_agent_->SetTrainingMode(true);
    
    Logger::Log(LogLevel::INFO, "Starting self-play phase");
    self_play_buffer_.clear();
    
    for (int game = 0; game < config_.num_self_play_games; ++game) {
        if (game % 10 == 0) {
            Logger::Log(LogLevel::INFO, "Self-play progress: " + 
                std::to_string(game) + "/" + 
                std::to_string(config_.num_self_play_games));
        }
        
        Logger::Log(LogLevel::DEBUG, "Starting game " + std::to_string(game));
        auto start_time = std::chrono::steady_clock::now();
        
        auto result = arena_.PlayGame(training_agent_, training_agent_, 
                                    initial_state_, true);
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        Logger::Log(LogLevel::DEBUG, "Game " + std::to_string(game) + 
            " completed in " + std::to_string(duration) + " seconds");
        
        Logger::Log(LogLevel::DEBUG, "Game history size: " + 
            std::to_string(result.game_history.size()));
        
        self_play_buffer_.insert(
            self_play_buffer_.end(),
            result.game_history.begin(),
            result.game_history.end()
        );
    }
    
    Logger::Log(LogLevel::INFO, "Training network on " + 
        std::to_string(self_play_buffer_.size()) + " positions");
    
    auto mcts_agent = std::dynamic_pointer_cast<MCTSAgent>(training_agent_);
    if (mcts_agent) {
        mcts_agent->TrainOnBuffer(self_play_buffer_);
    }
    
    if (EvaluateNewNetwork()) {
        Logger::Log(LogLevel::INFO, "New network accepted as best");
        SaveCheckpoint(config_.checkpoint_dir + "/best_network.pt");
        auto training_mcts = std::dynamic_pointer_cast<MCTSAgent>(training_agent_);
        auto network = std::dynamic_pointer_cast<ValuePolicyNetwork>(training_mcts->GetNetwork()->clone());
        best_agent_ = std::make_shared<MCTSAgent>(network, config_);
    } else {
        Logger::Log(LogLevel::INFO, "New network rejected, reverting");
        auto best_mcts = std::dynamic_pointer_cast<MCTSAgent>(best_agent_);
        auto network = std::dynamic_pointer_cast<ValuePolicyNetwork>(best_mcts->GetNetwork()->clone());
        training_agent_ = std::make_shared<MCTSAgent>(network, config_);
    }
}

bool TrainingManager::EvaluateNewNetwork() {
    training_agent_->SetTrainingMode(false);
    best_agent_->SetTrainingMode(false);
    
    Logger::Log(LogLevel::INFO, "Evaluating new network");
    
    int wins = 0, losses = 0, draws = 0;
    
    // Play games with training agent as both first and second player
    for (int game = 0; game < config_.games_per_evaluation; ++game) {
        bool training_agent_is_first = (game % 2 == 0);
        auto result = arena_.PlayGame(
            training_agent_is_first ? training_agent_ : best_agent_,
            training_agent_is_first ? best_agent_ : training_agent_,
            initial_state_, 
            false
        );
        
        // Adjust result based on who played first
        int adjusted_result = training_agent_is_first ? result.winner : -result.winner;
        
        if (adjusted_result == 1) {
            wins++;
            Logger::Log(LogLevel::INFO, "Game " + std::to_string(game) + ": Win");
        } else if (adjusted_result == -1) {
            losses++;
            Logger::Log(LogLevel::INFO, "Game " + std::to_string(game) + ": Loss");
        } else {
            draws++;
            Logger::Log(LogLevel::INFO, "Game " + std::to_string(game) + ": Draw");
        }
    }
    
    double win_rate = (wins + 0.5 * draws) / config_.games_per_evaluation;
    Logger::Log(LogLevel::INFO, "Evaluation results - Wins: " + std::to_string(wins) + 
                               ", Losses: " + std::to_string(losses) + 
                               ", Draws: " + std::to_string(draws));
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