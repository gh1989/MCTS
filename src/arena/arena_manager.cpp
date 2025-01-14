#include "arena/arena_manager.h"
#include "common/logger.h"
#include <cmath>
#include <random>

MatchResult ArenaManager::PlayGame(std::shared_ptr<Agent> player1,
                                 std::shared_ptr<Agent> player2,
                                 std::shared_ptr<State> initial_state,
                                 bool record_history) {
    MatchResult result;
    result.player1 = player1;
    result.player2 = player2;
    auto state = std::shared_ptr<State>(initial_state->Clone());
    std::shared_ptr<Agent> current_player = player1;
    
    std::vector<std::pair<std::shared_ptr<State>, int>> temp_history;
    
    Logger::Log(LogLevel::DEBUG, "Starting new game with history recording: " + 
                std::to_string(record_history));
    
    // Store initial state if recording
    if (record_history) {
        Logger::Log(LogLevel::DEBUG, "Recording initial state");
        temp_history.emplace_back(std::shared_ptr<State>(state->Clone()), 0);
    }
    
    if (!record_history) {
        Logger::Log(LogLevel::INFO, "\n=== New Game ===");
        state->Print();
    }
    
    int move_count = 0;
    while (!state->IsTerminal()) {
        Logger::Log(LogLevel::DEBUG, "Move " + std::to_string(move_count) + 
                   ", Current player: " + std::to_string(state->GetCurrentPlayer()));
        
        // Get action
        int action = current_player->GetAction(state);
        Logger::Log(LogLevel::DEBUG, "Selected action: " + std::to_string(action));
        
        if (action == -1) {
            Logger::Log(LogLevel::ERROR, "Invalid action -1 returned by agent");
            break;
        }
        
        // Apply action
        state->ApplyAction(action);
        move_count++;
        
        // Record state AFTER applying action
        if (record_history) {
            auto state_copy = std::shared_ptr<State>(state->Clone());
            temp_history.emplace_back(state_copy, 0);
            Logger::Log(LogLevel::DEBUG, "Recorded state after move " + 
                       std::to_string(move_count));
        }
        
        if (!record_history) {
            state->Print();
        }
        
        current_player = (current_player == player1) ? player2 : player1;
    }
    
    // Calculate final result
    result.winner = state->Evaluate();
    
    // Update history with final outcomes
    if (record_history) {
        for (auto& [stored_state, outcome] : temp_history) {
            bool is_player_one = stored_state->GetCurrentPlayer() == 1;
            outcome = is_player_one ? result.winner : -result.winner;
        }
        result.game_history = std::move(temp_history);
    }
    
    if (!record_history) {
        Logger::Log(LogLevel::INFO, 
            std::string("Game Result: ") +
            (result.winner == 1 ? "Player 1 wins" : 
             result.winner == -1 ? "Player 2 wins" : "Draw"));
        Logger::Log(LogLevel::INFO, "=== Game End ===\n");
    }
    
    return result;
}

std::vector<MatchResult> ArenaManager::RunTournament(
    const std::vector<std::shared_ptr<Agent>>& agents,
    std::shared_ptr<State> initial_state,
    int games_per_matchup) {
    
    std::vector<MatchResult> results;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < agents.size(); ++i) {
        for (size_t j = i + 1; j < agents.size(); ++j) {
            Logger::Log(LogLevel::INFO, 
                "Playing matchup " + std::to_string(i) + " vs " + std::to_string(j));
            
            for (int game = 0; game < games_per_matchup; ++game) {
                // Generate a unique seed for this game
                unsigned int game_seed = rd();
                
                // Set the seed for both agents
                agents[i]->SetSeed(game_seed);
                agents[j]->SetSeed(game_seed ^ 0xFFFFFFFF);  // Different but deterministic seed for opponent
                
                bool i_plays_first = std::bernoulli_distribution(0.5)(gen);
                
                auto result = i_plays_first
                    ? PlayGame(agents[i], agents[j], initial_state, false)
                    : PlayGame(agents[j], agents[i], initial_state, false);
                    
                if (!i_plays_first) {
                    result.winner = -result.winner;
                }
                
                results.push_back(result);
            }
        }
    }
    
    return results;
}

void ArenaManager::UpdateRatings(const MatchResult& result) {
    // Get expected scores using ELO formula
    double player1_rating = GetAgentRating(result.player1);
    double player2_rating = GetAgentRating(result.player2);
    
    double rating_diff = (player2_rating - player1_rating) / 400.0;
    double expected_score1 = 1.0 / (1.0 + std::pow(10, rating_diff));
    double expected_score2 = 1.0 - expected_score1;
    
    // Calculate actual scores
    double actual_score1 = (result.winner == 1) ? 1.0 : 
                          (result.winner == 0) ? 0.5 : 0.0;
    double actual_score2 = 1.0 - actual_score1;
    
    // Update ratings
    double p1_change = kEloK * (actual_score1 - expected_score1);
    double p2_change = kEloK * (actual_score2 - expected_score2);
    
    elo_ratings_[result.player1] += p1_change;
    elo_ratings_[result.player2] += p2_change;
    
    // Store changes in the result (cast away const)
    const_cast<MatchResult&>(result).player1_elo_change = p1_change;
    const_cast<MatchResult&>(result).player2_elo_change = p2_change;
}

double ArenaManager::GetAgentRating(const std::shared_ptr<Agent>& agent) const {
    auto it = elo_ratings_.find(agent);
    return (it != elo_ratings_.end()) ? it->second : kInitialElo;
} 