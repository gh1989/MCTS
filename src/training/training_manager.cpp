#include "training/training_manager.h"
#include "common/logger.h"
#include "common/network.h"
#include "agents/mcts_agent.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <iomanip>

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
    
    Logger::Log(LogLevel::INFO, "Self-play games: 0/" + std::to_string(config_.num_self_play_games));
    
    self_play_buffer_.clear();
    
    auto opponent_agent = std::make_shared<MCTSAgent>(
        std::dynamic_pointer_cast<ValuePolicyNetwork>(
            std::dynamic_pointer_cast<MCTSAgent>(training_agent_)->GetNetwork()->clone()
        ), 
        config_
    );
    opponent_agent->SetTrainingMode(false);
    
    for (int game = 0; game < config_.num_self_play_games; ++game) {
        auto start_time = std::chrono::steady_clock::now();
        
        bool training_agent_first = (game % 2 == 0);
        auto first_player = training_agent_first ? training_agent_ : opponent_agent;
        auto second_player = training_agent_first ? opponent_agent : training_agent_;
        
        auto result = arena_.PlayGame(first_player, second_player, initial_state_, true);
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count() / 1000.0;
            
        std::cout << "\rSelf-play games: " << (game + 1) << "/" 
                  << config_.num_self_play_games << " (Time: " 
                  << std::fixed << std::setprecision(1) << duration 
                  << "s, Buffer: " << self_play_buffer_.size() << ")" << std::flush;
        
        self_play_buffer_.insert(
            self_play_buffer_.end(),
            result.game_history.begin(),
            result.game_history.end()
        );
        
        if (game % 10 == 0 && game > 0) {
            opponent_agent = std::make_shared<MCTSAgent>(
                std::dynamic_pointer_cast<ValuePolicyNetwork>(
                    std::dynamic_pointer_cast<MCTSAgent>(training_agent_)->GetNetwork()->clone()
                ), 
                config_
            );
            opponent_agent->SetTrainingMode(false);
        }
    }
    std::cout << std::endl;
    
    Logger::Log(LogLevel::INFO, "Training on " + std::to_string(self_play_buffer_.size()) + " positions");
    
    auto mcts_agent = std::dynamic_pointer_cast<MCTSAgent>(training_agent_);
    if (mcts_agent) {
        mcts_agent->TrainOnBuffer(self_play_buffer_);
    }
    
    Logger::Log(LogLevel::INFO, "Starting evaluation (" + 
                std::to_string(config_.games_per_evaluation) + " games)...");
    
    bool network_accepted = EvaluateNewNetwork();
    if (network_accepted) {
        Logger::Log(LogLevel::INFO, "✓ Network ACCEPTED - New best network saved");
        SaveCheckpoint(config_.checkpoint_dir + "/best_network.pt");
        auto training_mcts = std::dynamic_pointer_cast<MCTSAgent>(training_agent_);
        auto network = std::dynamic_pointer_cast<ValuePolicyNetwork>(training_mcts->GetNetwork()->clone());
        best_agent_ = std::make_shared<MCTSAgent>(network, config_);
    } else {
        Logger::Log(LogLevel::INFO, "✗ Network REJECTED - Reverting to previous best");
        auto best_mcts = std::dynamic_pointer_cast<MCTSAgent>(best_agent_);
        auto network = std::dynamic_pointer_cast<ValuePolicyNetwork>(best_mcts->GetNetwork()->clone());
        training_agent_ = std::make_shared<MCTSAgent>(network, config_);
    }
}

bool TrainingManager::EvaluateNewNetwork() {
    training_agent_->SetTrainingMode(false);
    best_agent_->SetTrainingMode(false);
    
    int wins = 0, losses = 0, draws = 0;
    
    for (int game = 0; game < config_.games_per_evaluation; ++game) {
        bool training_agent_is_first = (game % 2 == 0);
        auto result = arena_.PlayGame(
            training_agent_is_first ? training_agent_ : best_agent_,
            training_agent_is_first ? best_agent_ : training_agent_,
            initial_state_, 
            false
        );
        
        int adjusted_result = training_agent_is_first ? result.winner : -result.winner;
        
        if (adjusted_result == 1) wins++;
        else if (adjusted_result == -1) losses++;
        else draws++;
        
        if (progress_callback_) {
            std::string status = "W: " + std::to_string(wins) + 
                               " L: " + std::to_string(losses) + 
                               " D: " + std::to_string(draws) + 
                               " | Win rate: " + 
                               std::to_string((wins + 0.5 * draws) / (game + 1));
            progress_callback_("Evaluation", game + 1, config_.games_per_evaluation, status);
        }
    }
    
    double win_rate = (wins + 0.5 * draws) / config_.games_per_evaluation;
    return win_rate >= config_.required_win_rate;
}

void TrainingManager::SaveCheckpoint(const std::string& filepath) {
    training_agent_->SaveModel(filepath);
}

void TrainingManager::LoadCheckpoint(const std::string& filepath) {
    training_agent_->LoadModel(filepath);
    best_agent_->LoadModel(filepath);
} 