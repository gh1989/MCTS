#include "training/training_manager.h"
#include "common/logger.h"
#include "common/network.h"
#include "agents/mcts_agent.h"
#include "agents/random_agent.h"
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
      arena_(initial_state),
      replay_buffer_(config_.replay_buffer_size) {
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
    // Set higher temperature for training
    config_.temperature = 1.0;
    training_agent_->SetTrainingMode(true);
    
    Logger::Log(LogLevel::INFO, "Self-play games: 0/" + std::to_string(config_.num_self_play_games));
    
    for (int game = 0; game < config_.num_self_play_games; ++game) {
        auto start_time = std::chrono::steady_clock::now();
        
        bool training_agent_first = (game % 2 == 0);
        auto first_player = training_agent_first ? training_agent_ : best_agent_;
        auto second_player = training_agent_first ? best_agent_ : training_agent_;
        
        auto result = arena_.PlayGame(first_player, second_player, initial_state_, true);
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count() / 1000.0;
            
        for (const auto& [state, outcome] : result.game_history) {
            replay_buffer_.Add(state->Clone(), outcome);
        }
        
        std::cout << "\rSelf-play games: " << (game + 1) << "/" 
                  << config_.num_self_play_games << " (Time: " 
                  << std::fixed << std::setprecision(1) << duration 
                  << "s, Buffer: " << replay_buffer_.Size() << ")" << std::flush;
    }
    std::cout << std::endl;
    
    Logger::Log(LogLevel::INFO, "Training on " + std::to_string(replay_buffer_.Size()) + " positions");
    
    auto mcts_agent = std::dynamic_pointer_cast<MCTSAgent>(training_agent_);
    if (mcts_agent) {
        mcts_agent->TrainOnBuffer(replay_buffer_);
    }
    
    // During evaluation, use temperature = 0
    config_.temperature = 0.0;
    training_agent_->SetTrainingMode(false);
    
    int wins = 0;
    auto random_agent = std::make_shared<RandomAgent>();
    
    for (int i = 0; i < config_.games_per_evaluation; ++i) {
        auto result = arena_.PlayGame(training_agent_, random_agent, initial_state_, false);
        if (result.winner == 1) {
            wins++;
        }
    }
    
    double win_rate = static_cast<double>(wins) / config_.games_per_evaluation;
    Logger::Log(LogLevel::INFO, 
        "Evaluation complete - Win rate against random: " + 
        std::to_string(win_rate));
    
    // Always save the latest network
    SaveCheckpoint(config_.checkpoint_dir + "/best_network.pt");
    
    // Reset temperature for next training iteration
    config_.temperature = 1.0;
}

bool TrainingManager::EvaluateNewNetwork() {
    training_agent_->SetTrainingMode(false);
    best_agent_->SetTrainingMode(false);
    
    int wins = 0, losses = 0, draws = 0;
    double total_game_length = 0;
    
    Logger::Log(LogLevel::INFO, "Evaluating against previous best network...");
    
    for (int game = 0; game < config_.games_per_evaluation; ++game) {
        bool training_agent_is_first = (game % 2 == 0);
        auto result = arena_.PlayGame(
            training_agent_is_first ? training_agent_ : best_agent_,
            training_agent_is_first ? best_agent_ : training_agent_,
            initial_state_, 
            true  // Record history to analyze game length
        );
        
        int adjusted_result = training_agent_is_first ? result.winner : -result.winner;
        if (adjusted_result == 1) wins++;
        else if (adjusted_result == -1) losses++;
        else draws++;
        
        total_game_length += result.game_history.size();
        
        if (progress_callback_) {
            double current_win_rate = (wins + 0.5 * draws) / (game + 1);
            std::string status = "W: " + std::to_string(wins) + 
                               " L: " + std::to_string(losses) + 
                               " D: " + std::to_string(draws) + 
                               " | Win rate: " + std::to_string(current_win_rate);
            progress_callback_("Evaluation", game + 1, config_.games_per_evaluation, status);
        }
    }
    
    double win_rate = (wins + 0.5 * draws) / config_.games_per_evaluation;
    double avg_game_length = total_game_length / config_.games_per_evaluation;
    
    Logger::Log(LogLevel::INFO, "Evaluation Results:");
    Logger::Log(LogLevel::INFO, "Win Rate: " + std::to_string(win_rate));
    Logger::Log(LogLevel::INFO, "Average Game Length: " + std::to_string(avg_game_length));
    
    // Store metrics for tracking improvement
    evaluation_history_.push_back({win_rate, avg_game_length});
    
    // Check if performance is improving
    bool is_improving = true;
    if (evaluation_history_.size() > 1) {
        const auto& prev = evaluation_history_[evaluation_history_.size() - 2];
        is_improving = (win_rate > prev.win_rate) || 
                      (win_rate == prev.win_rate && avg_game_length < prev.avg_game_length);
    }
    
    return win_rate >= config_.required_win_rate && is_improving;
}

void TrainingManager::SaveCheckpoint(const std::string& filepath) {
    training_agent_->SaveModel(filepath);
}

void TrainingManager::LoadCheckpoint(const std::string& filepath) {
    training_agent_->LoadModel(filepath);
    best_agent_->LoadModel(filepath);
} 