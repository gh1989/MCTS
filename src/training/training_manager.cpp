#include "training/training_manager.h"
#include "common/logger.h"
#include "common/network.h"
#include "agents/mcts_agent.h"
#include "networks/tic_tac_toe_network.h"
#include <filesystem>
#include <algorithm>

TrainingManager::TrainingManager(const TrainingConfig& config,
                               std::shared_ptr<State> initial_state,
                               std::shared_ptr<ValuePolicyNetwork> network)
    : config_(config),
      initial_state_(initial_state),
      arena_() {
    std::filesystem::create_directories(config.checkpoint_dir);
    
    auto value_policy_net = std::dynamic_pointer_cast<ValuePolicyNetwork>(network);
    if (!value_policy_net) {
        throw std::runtime_error("Network must be a ValuePolicyNetwork");
    }
    
    best_agent_ = std::make_shared<MCTSAgent>(
        std::dynamic_pointer_cast<ValuePolicyNetwork>(network), 
        config
    );
    training_agent_ = std::make_shared<MCTSAgent>(
        std::dynamic_pointer_cast<ValuePolicyNetwork>(network)->clone(),
        config
    );
}

void TrainingManager::RunTrainingIteration() {
    Logger::Log(LogLevel::INFO, "Starting self-play phase");
    self_play_buffer_.clear();
    
    for (int game = 0; game < config_.num_self_play_games; ++game) {
        auto result = arena_.PlayGame(training_agent_, training_agent_, 
                                    initial_state_, true);
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
        best_agent_ = std::make_shared<MCTSAgent>(
            std::dynamic_pointer_cast<MCTSAgent>(training_agent_)->CloneNetwork(),
            config_
        );
    } else {
        Logger::Log(LogLevel::INFO, "New network rejected, reverting");
        training_agent_ = std::make_shared<MCTSAgent>(
            std::dynamic_pointer_cast<MCTSAgent>(best_agent_)->CloneNetwork(),
            config_
        );
    }
}

bool TrainingManager::EvaluateNewNetwork() {
    Logger::Log(LogLevel::INFO, "Evaluating new network");
    
    int wins = 0, losses = 0, draws = 0;
    
    for (int game = 0; game < config_.games_per_evaluation; ++game) {
        auto result = (game % 2 == 0)
            ? arena_.PlayGame(training_agent_, best_agent_, false)
            : arena_.PlayGame(best_agent_, training_agent_, false);
            
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